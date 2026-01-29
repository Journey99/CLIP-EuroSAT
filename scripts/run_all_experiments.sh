#!/bin/bash

echo "Running all CLIP-EuroSAT experiments..."

# Zero-shot
echo "1/4: Running zero-shot experiment..."
python experiments/zero_shot.py

# Few-shot Linear Probe
echo "2/4: Running few-shot linear probe..."
python experiments/few_shot_linear.py

# Few-shot LoRA
echo "3/4: Running few-shot LoRA..."
python experiments/few_shot_lora.py

# Full Fine-tuning
echo "4/4: Running full fine-tuning..."
python experiments/full_finetune.py

echo "✓ All experiments completed!"
echo "Results saved to results/"

