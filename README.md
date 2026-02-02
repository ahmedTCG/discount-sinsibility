# Discount Sensitivity Pipeline

This repository contains a reproducible machine-learning pipeline to identify and segment customers by discount sensitivity for marketing use.

## Goal

Classify customers into three segments:
- **full_price** – rarely need discounts
- **conditional** – respond selectively
- **discount_driven** – strongly motivated by discounts

## Repository Structure

- scripts/config.py – Central configuration
- scripts/01_prepare_data.py – Data preparation and target creation
- scripts/02_train_model.py – Model training (LightGBM)
- scripts/03_score.py – Customer scoring
- scripts/04_bucketize.py – Segment assignment
- artifacts/ – Model, metadata, metrics
- data/ – Cleaned data and outputs
- run_all.py – End-to-end pipeline runner

## Run the Pipeline

python run_all.py --input path/to/raw_input.csv

Example:
python run_all.py --input discount_sensitivity_features_v10.csv

To skip training and use an existing model:
python run_all.py --input path/to/raw_input.csv --skip-train

## Output

Final file for Marketing: data/scores_bucketized.csv

Columns:
- externalcustomerkey – Customer identifier
- score – Discount sensitivity probability (0-1)
- segment – Assigned segment

## Model Performance

- Validation ROC-AUC: 0.911
- Training samples: 4,658,592
- Validation samples: 1,164,649
- Features: 44

## Segment Thresholds

- full_price: 0.00 - 0.20 (Low discount sensitivity)
- conditional: 0.20 - 0.50 (Moderate discount sensitivity)
- discount_driven: 0.50 - 1.00 (High discount sensitivity)

## Segment Distribution

- full_price: 3,736,541 (64.2%)
- conditional: 951,070 (16.3%)
- discount_driven: 1,135,633 (19.5%)

## Data Leakage Safety

- Discount-related variables removed before training
- Feature schema enforced
- Identical preprocessing logic for training and scoring

## Requirements

See requirements.txt for dependencies.
