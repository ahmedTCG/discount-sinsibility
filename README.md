# Discount Sensitivity Pipeline

This repository contains a reproducible machine-learning pipeline to identify and segment customers by discount sensitivity for marketing use.

## Goal

Classify customers into three segments:
- full_price – rarely need discounts
- conditional – respond selectively
- discount_driven – strongly motivated by discounts

## Repository Structure

- scripts/01_prepare_data.py – data preparation and target creation
- scripts/02_train_model.py – model training (LightGBM)
- scripts/03_score.py – customer scoring
- scripts/04_bucketize.py – segment assignment
- run_all.py – end-to-end pipeline runner

Data, models, and outputs are intentionally not tracked in Git.

## Run the Pipeline

python run_all.py --input path/to/raw_input.csv

Example:
python run_all.py --input discount_sensitivity_features_fulldata.csv

## Output

Final file to share with Marketing:
data/marketing_discount_segments.csv

Columns:
- externalcustomerkey
- segment (full_price, conditional, discount_driven)

## Segment Thresholds

full_price:      0.00 – 0.20  
conditional:     0.20 – 0.60  
discount_driven: 0.60 – 1.00  

## Data Leakage Safety

- Discount-related variables removed
- Feature schema enforced
- Identical logic for training and scoring

Model performance: ROC-AUC ≈ 0.97


