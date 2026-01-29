# CLIP-EuroSAT: Zero-shot to Fine-tuning Analysis

## 실험 목표
> CLIP 기반 vision foundation model에 대해 zero-shot / few-shot / LoRA fine-tuning / full fine-tuning 성능을 비교 분석

## EuroSAT 데이터셋
- Total Number of Images: 27000
- Bands: 3 (RGB)
- Image Resolution: 64x64m
- Land Cover Classes: 10

## 실험 구성
1. **Zero-shot**: Prompt engineering만으로 분류
2. **Few-shot (Linear Probe)**: 소량 데이터로 선형 분류기 학습
3. **Few-shot (LoRA)**: LoRA로 효율적 fine-tuning
4. **Full Fine-tuning**: 전체 모델 학습

## Quick Start
```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 다운로드
bash scripts/download_data.sh

# 3. 모든 실험 실행
bash scripts/run_all_experiments.sh

# 4. 결과 확인
jupyter notebook notebooks/04_analysis.ipynb
```

## 🧪 실험 단계별 설계
### Experiment 0 — Baseline 정보
| 항목         | 내용                 |
| ---------- | ------------------ |
| Model      | CLIP ViT-B/32      |
| Image size | 224×224            |
| Dataset    | EuroSAT (10 classes) |
| Split      | Train 70% / Val 20% / Test 10% |
| Metric     | Accuracy, Precision, Recall, F1 (Macro) |
| Seed       | 42       |


### Experiment 1 — Zero-shot Classification
| 항목     | 내용                               |
| ------ | -------------------------------- |
| 학습     | 없음 (0 epochs)                   |
| Prompt Templates | 8가지 ensemble (e.g., "a centered satellite photo of {class}") |
| CLIP encoder | frozen (pretrained) |
| 평가     | test set 전체                      |
| 목적     | Pretrained CLIP의 zero-shot 능력 측정 |
| Batch size | 128 |


### Experiment 2 — Few-shot Linear Probe
| 항목           | 내용                   |
| ------------ | -------------------- |
| Shot         | k ∈ {1, 5, 10, 20} shots/class |
| 학습 파라미터   | Linear classifier head only (~10K params) |
| CLIP encoder | frozen               |
| Epochs | 100 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Batch size | 32 |
| 목적      | CLIP features의 task 적응력 평가 |


### Experiment 3 — Few-shot LoRA Fine-tuning
| 항목           | 내용                             |
| ------------ | ------------------------------ |
| Shot         | k ∈ {1, 5, 10, 20} shots/class |
| 튜닝 방식      | LoRA (ViT attention layers)    |
| LoRA Rank    | r = 8                          |
| LoRA Alpha   | α = 16                         |
| Dropout      | 0.1                            |
| Target modules | q_proj, v_proj, out_proj     |
| Image encoder | LoRA adapters only (~0.3M params, 0.4%) |
| Text encoder | frozen                         |
| Epochs | 50 |
| Learning rate | 5e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| 목적           | Parameter-efficient adaptation 효율성 검증 |


### Experiment 4 — Full Fine-tuning
| 항목   | 내용               |
| ---- | ---------------- |
| Shot | Full training set (70%) |
| 튜닝   | Image encoder 전체 (~87M params) |
| Text encoder | frozen |
| Epochs | 30 |
| Learning rate | 1e-5 (낮은 lr) |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| 목적   | 성능 upper bound 측정 |


## 🔬 분석 포인트

### 1. Zero-shot Performance
- Prompt engineering 효과 (simple vs ensemble templates)
- Domain shift 정량화 (ImageNet → Satellite)

### 2. Few-shot Learning Curve
- Shot 수 증가에 따른 성능 향상 (1 → 5 → 10 → 20)
- 어떤 클래스가 few-shot에 강한가?

### 3. LoRA Efficiency
- Linear Probe vs LoRA (같은 shot 수)
- Parameter 대비 성능 (0.4% params로 얼마나?)
- Rank(r) 영향 분석 (optional: r=4, 8, 16 비교)

### 4. Full Fine-tuning Analysis
- Overfitting 여부 (train/val/test gap)
- LoRA vs Full FT 성능 차이
- 학습 곡선 비교 (수렴 속도)

### 5. Per-class Performance
- Confusion matrix로 어려운 클래스 쌍 파악
- 각 method별 강점/약점 클래스 분석


## 📁 프로젝트 구조

- `data/`: EuroSAT 데이터셋
- `experiments/`: 각 실험 스크립트
- `lora/`: LoRA 구현
- `models/`: CLIP wrapper
- `utils/`: 유틸리티 함수
- `results/`: 실험 결과