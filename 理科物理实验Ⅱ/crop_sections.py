"""按小节从扫描版实验教材 PDF 中自动截取内容图片。

用法:
    python crop_sections.py 5.10              # 截取实验 5.10, 写入 "5.10.../<小节>/"
    python crop_sections.py 5.10 5.3          # 批量
    python crop_sections.py                   # 处理所有 "编号+名称" 目录
    python crop_sections.py 5.3 --dpi 200     # 更高分辨率
    python crop_sections.py 5.10 --list       # 只列出识别到的小节

依赖: pymupdf, pillow, numpy, rapidocr_onnxruntime (pip install 即可)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "理科物理实验教程 力学 热学 电磁学 光学实验分册.pdf"

SECTIONS = [
    "引言", "实验目的", "预习提示", "实验仪器及样品", "实验原理",
    "实验内容与测量", "实验内容", "实验步骤", "测量原理", "数据处理",
    "注意事项", "讨论与思考题", "讨论", "结论", "研究性题目", "思考题",
    "参考文献",
]

X0, X1 = 0.105, 0.905         # 正文裁剪的左右边界(占页宽比例), 去掉虚线和页码
HEADER = 0.075                 # 页眉高度比例
PAD = 10                       # 小节边界上下留白(像素)


def clean(text):
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def match_heading(text):
    s = clean(text)
    for name in SECTIONS:
        if name in s and len(s) <= len(name) + 3:
            return name
    return None


def experiment_pages(doc, number):
    toc = [(t, p) for _, t, p in doc.get_toc()]
    pat = re.compile(r"^实验\s*" + re.escape(number) + r"(?![\d.])")
    starts = [p for t, p in toc if pat.match(t.replace("\u3000", " ").strip())]
    if not starts:
        sys.exit(f"目录中找不到实验 {number}")
    first = starts[0]
    later = [p for _, p in toc if p > first]
    last = min(later) - 1 if later else doc.page_count
    return first, last


def ink_box(img):
    rows = np.where((img < 160).sum(1) > 2)[0]
    if not len(rows):
        return None
    return rows.min(), rows.max()


def scan_page(doc, ocr, pno, dpi):
    pix = doc[pno].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = np.array(Image.fromarray(img).convert("L"))
    result, _ = ocr(gray)
    heads = []
    for box, text, _score in result or []:
        name = match_heading(text)
        if name:
            heads.append((round(box[0][1]), name))
    return gray, sorted(heads)


def crop_band(gray, y0, y1):
    h, w = gray.shape
    x0, x1 = int(w * X0), int(w * X1)
    y0, y1 = max(int(h * HEADER), y0), min(h, y1)
    band = gray[y0:y1, x0:x1]
    box = ink_box(band)
    if box is None:
        return None
    top = max(0, box[0] - PAD // 2)
    bottom = min(band.shape[0], box[1] + PAD // 2)
    return band[top:bottom]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="*", help="实验编号, 如 5.10 5.3; 省略则处理所有编号目录")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--list", action="store_true", help="只打印识别到的小节")
    args = ap.parse_args()

    numbers = args.number
    if not numbers:
        numbers = [m.group(1) for p in ROOT.iterdir() if p.is_dir()
                   for m in [re.match(r"^(\d+\.\d+)(?![\d.])", p.name)] if m]
    if not numbers:
        sys.exit("没有找到实验编号目录")

    doc = pymupdf.open(PDF)
    ocr = RapidOCR()
    for number in numbers:
        process(doc, ocr, number, args)


def process(doc, ocr, number, args):
    first, last = experiment_pages(doc, number)
    print(f"实验 {number}: PDF 第 {first}-{last} 页")

    out_root = next((p for p in ROOT.iterdir()
                     if p.is_dir() and re.match(re.escape(number) + r"(?![\d.])", p.name)), None)
    out_root = out_root or ROOT / f"实验{number}"

    pages = {}
    marks = []
    for pno in range(first - 1, last):
        gray, heads = scan_page(doc, ocr, pno, args.dpi)
        pages[pno] = gray
        marks += [(pno, y, name) for y, name in heads]
        print(f"  第 {pno + 1} 页: {[n for _, n in heads]}")

    marks.sort()
    lead = next((i for i, (_, _, n) in enumerate(marks) if n in ("引言", "实验目的")), None)
    if lead is not None:
        marks = marks[lead:]
    if len(marks) < 2:
        sys.exit("识别到的小节不足 2 个, 请检查 --dpi 或该实验的排版")

    for i, (pno, y, name) in enumerate(marks):
        nxt = marks[i + 1] if i + 1 < len(marks) else None
        pieces = []
        for p in range(pno, (nxt[0] if nxt else pno) + 1):
            gray = pages[p]
            top = y if p == pno else int(gray.shape[0] * HEADER)
            bottom = nxt[1] if (nxt and p == nxt[0]) else gray.shape[0]
            band = crop_band(gray, top, bottom)
            if band is not None and band.shape[0] > 40:
                pieces.append((p, band))
        if not pieces:
            continue
        dest = out_root / name
        print(f"[{name}] PDF 第 {pieces[0][0] + 1}-{pieces[-1][0] + 1} 页, {len(pieces)} 张")
        if args.list:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for idx, (p, band) in enumerate(pieces, 1):
            Image.fromarray(band).save(dest / f"{idx:02d}_p{p + 1}.png")


if __name__ == "__main__":
    main()
