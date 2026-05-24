"""eelsSaver pipeline package.

Stages:
    s2_features  — Sentinel-2 rolling temporal features per pixel.
    teacher      — ODE-based GPI + decline + stress + SHAP soft labels.
    student      — XGBoost models trained on Forillon pixel-window data.
    inference    — Pan-Atlantic scoring + bed aggregation + OOD detection.
    calibration  — Optional ground-truth recalibration.
    run_all      — Orchestrates the training pipeline.
"""
