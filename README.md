# Phishing Detection with Concept Drift Analysis

## MSc Data Science Project — Coventry University 2026

## Project Overview
This project develops an adaptive phishing URL detection system 
that compares XGBoost and Random Forest classifiers under concept 
drift conditions. The Kolmogorov-Smirnov test is used to detect 
distributional shift between training and incoming data, and 
adaptive retraining is applied when significant drift is detected.

## Files in this Repository
- `dashboard.py` — Interactive Streamlit dashboard
- `phishing xgboost and randomforest.ipynb` — Model training, 
   evaluation and drift analysis notebook
- `phishing_model_retrained.pkl` — Trained XGBoost model
- `rf_model_retrained.pkl` — Trained Random Forest model

## Dataset
Dataset sourced from Mendeley Data (Vrbančič, 2020):
https://doi.org/10.17632/72ptz43s9v.1

- dataset_full.csv — 88,647 records (training baseline)
- dataset_small.cs
