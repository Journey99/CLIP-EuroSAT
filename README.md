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
- Rank(r) 영향 분석 (optional: r=4, 8 비교)

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


## 결과 분석
| 방법               | accuracy|
| ---------------- | ---------------- |
| Zero-shot CLIP   | 41.6%        |
| Few-shot Linear (1/5/10/20)  | 59.6 % / 76.1% / 76.8% / 80.4% |
| LoRA (1/5/10/20)    | 44.5% / 46.9% / 72.9% / 75.03%   |
| Full fine-tuning | 98.5%          |

### 1. Zero-shot clip
EuroSAT은 satellite imagery인데 CLIP는 internet image + text로 학습되었다. 그래서 domain gap이 생기고 40~60% 정도의 정확도가 나올 수 있다.

### 2. Few-shot Linear
이 패턴은 CLIP few-shot 논문에서도 거의 동일하게 나타난다.
1. CLIP feature quality
    - CLIP image encoder의 embedding은 이미 semantic feature space를 형성
    - 즉 eature space에서 forest/river 같은 클래스가 어느 정도 분리
    - 그래서 linear classifier는 단순히 decision boundary만 학습하면 됨
2. 파마리터 수가 매우 적음
    - linear probe에서 학습되는 것은 embedding_dim × num_classes
    - 그래서 few-shot에서도 안정적으로 학습 가능

### 3. Few-shot LoRA
1,5-shot → 거의 zero-shot 수준 / 10-shot부터 급상승 이 패턴은 흔한 패턴이다.
1. low-shot에서 발생하는 문제
예를 들어 1-shot이면 train data = 10 / train params = 300k 이다. 이 경우 gradient noise, optimization instablility가 발생한다. 그래서 representation adaptation이 제대로 일어나지 않아 zero-shot과 비슷한 정확도가 나올 수 있다.
2. 10-shot에서 올라가는 이유
LoRA는 backbone feature를 수정한다. 즉, representation learning 문제다. representation을 수정하려면 decision boundary 학습보다 훨씬 많은 데이터가 필요하다. 그래서 10-shot 부터는 representation gradient signal이 충분해진다. 그래서 정확도가 급 상승할 수 있다.
3. Linear vs LoRA 결과 차이의 본질

    | 방법           | 학습 대상                   |
    | ------------ | ----------------------- |
    | Linear probe | classifier              |
    | LoRA         | backbone representation |
4. LoRA가 20-shot에서도 linear보다 낮은 이유
    1. training epoch 차이
        - representation tuning은 더 오래 학습해야함
    2. learning rate
        - 높은 학습률로 1e-4 ~ 2e-4 낮추면 개선 가능
    3. LoRA가 20-shot에서도 linear보다 낮은 이유
        - 현재 적용 위치에 k_proj + mlp 추가하면 성능 개선 가능
    4. Rank
        - few-shot에서는 r=4가 안정적인 경우도 많다.
        - rank 가 작을수록 파라미터가 작아지고 오버피팅이 감소

### 4. Full fine-tuning
- full_finetune.py
    - encoder와 classifier 모두 **아주 낮은 lr(1e-5)**로만 학습 → 초기 몇 epoch 동안 변화가 매우 느림
    - warmup/clip 없음 → 큰 문제는 아니지만 수렴 속도가 느리고, 특정 seed에서 평평하게 머무를 위험이 있음
- full_finetune_v2.py
    - warmup + cosine으로 안정적 시작
    - gradient clipping으로 폭주 방지

### 정리
- EuroSAT 데이터셋에서 Full Fine-tuning은 98%의 정확도를 달성하며 parameter-efficient 방법들보다 훨씬 높은 성능을 보였다.
- 이는 CLIP의 feature가 위성 이미지에서도 이미 어느 정도 구별력을 가지지만, backbone 전체를 업데이트할 경우 원격탐사 데이터에 특화된 표현(domain-specific representation)을 추가로 학습할 수 있기 때문으로 해석할 수 있다.
- 또한 few-shot adaptation과 full fine-tuning 사이의 큰 성능 차이는 효과적인 representation 학습을 위해 충분한 학습 데이터가 중요하다는 점을 보여준다.