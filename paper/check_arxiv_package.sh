#!/bin/bash
# Kiểm tra một gói .tar.gz có dịch được trên arXiv không.
#   ./check_arxiv_package.sh <goi.tar.gz> <ten_file_chinh_khong_duoi>
# arXiv chỉ có những file trong gói, và không chạy bibtex -- script này mô phỏng
# đúng hai điều kiện đó.
set -e
PKG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); MAIN=${2:-main}
DIR=$(mktemp -d)
tar xzf "$PKG" -C "$DIR"
cd "$DIR"
for i in 1 2 3; do pdflatex -interaction=nonstopmode "$MAIN.tex" >/dev/null 2>&1 || true; done
echo "thư mục trắng : $DIR"
echo "lỗi biên dịch : $(grep -cE '^! ' "$MAIN.log" 2>/dev/null || true)"
echo "citation hỏng : $(grep -c 'Citation.*undefined' "$MAIN.log" 2>/dev/null || true)"
echo "tham chiếu ?? : $(pdftotext "$MAIN.pdf" - 2>/dev/null | grep -o '??' | wc -l | tr -d ' ')"
echo "mục References: $(pdftotext "$MAIN.pdf" - 2>/dev/null | grep -c '^References' || true)"
echo "số trang      : $(pdfinfo "$MAIN.pdf" 2>/dev/null | awk '/Pages/{print $2}')"
