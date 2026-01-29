#!/bin/bash

set -e  # 중간에 에러 나면 즉시 중단

echo "Downloading EuroSAT dataset..."

mkdir -p data
cd data

# Kaggle에서 EuroSAT 다운로드
kaggle datasets download -d apollo2506/eurosat-dataset

# 압축 해제
unzip eurosat-dataset.zip
rm eurosat-dataset.zip

# 폴더 이름 정리
mv EuroSAT eurosat

# 멀티밴드 데이터 제거 (CLIP 실험에 불필요)
if [ -d "EuroSATallBands" ]; then
    echo "Removing EuroSATallBands (not needed for CLIP)..."
    rm -rf EuroSATallBands
fi

echo "✓ EuroSAT dataset prepared at data/eurosat/"

# 데이터 구조 확인
echo "Data structure:"
ls -l eurosat/

echo "Number of images per class:"
for class_dir in eurosat/*/; do
    count=$(ls -1 "$class_dir" | wc -l)
    echo "$(basename "$class_dir"): $count images"
done
