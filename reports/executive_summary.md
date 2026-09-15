# Executive Summary — House Price Prediction

## Business Objective

Build a machine learning pipeline that predicts residential property sale prices from structural and categorical features, enabling data-driven real estate valuation through an interactive web dashboard.

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | Ames Housing (1,460 records, 79 features + 6 engineered) |
| Models Evaluated | 5 (Linear, Ridge, Lasso, Random Forest, XGBoost) |
| Best CV RMSE | $30,356 (Ridge Regression) |
| Best Holdout R² | 0.888 (XGBoost) |
| Selected Deployment Model | **Ridge Regression** |
| Deployment RMSE Tolerance | 0.00% (best model is browser-compatible) |
| Browser Inference | Zero-dependency JavaScript (no backend) |
| Parity Test | 5/5 cases passed ($0.00 difference) |
| Diagnostic Tests | 15/15 passed |

## Architecture

```
Ames Housing Dataset
        ↓
Leakage-safe preprocessing (Pipeline + ColumnTransformer)
        ↓
6 engineered domain features
        ↓
5 regression models with 5-fold CV
        ↓
80/20 stratified holdout evaluation
        ↓
5% deployment-selection rule
        ↓
Ridge Regression → docs/model_export.json
        ↓
Browser-side JavaScript inference
        ↓
3-section executive dashboard (GitHub Pages)
```

## Deployment

- **Platform:** GitHub Pages (`/docs` directory)
- **Inference:** Client-side JavaScript, no backend required
- **Model artifact:** `docs/model_export.json` (single authoritative export)
- **Dashboard sections:** Property Valuation (with Valuation Recommendation), Try a Scenario (4 presets), Model Performance, Housing Insights

## Technical Highlights

- **Zero data leakage:** Preprocessing fitted only on training folds; holdout untouched until final scoring
- **Reproducible:** `random_state=42` for all stochastic operations
- **Parity-verified:** Python and JavaScript produce identical predictions ($0.00 difference)
- **Validated:** 15/15 diagnostic tests pass, CV and holdout independently verified
- **Executive-oriented:** Focused 3-section dashboard with KPI cards, interactive inputs, Chart.js visualizations, and valuation recommendation

## Validation Status: FINAL & FROZEN

> `Id` was retained in the source dataset for identification and diagnostics but excluded from the predictive feature matrix because it is an identifier rather than a meaningful property attribute. A controlled comparison confirmed improved CV performance across the evaluated models, with improvements ranging from approximately 0.3% to 2.19%.

## Files Delivered

| File | Purpose |
|------|---------|
| `main.py` | Single-command pipeline orchestrator |
| `docs/index.html` | Dashboard UI |
| `docs/app.js` | Client-side inference engine + valuation recommendation + demo scenarios |
| `docs/model_export.json` | Model artifact for browser |
| `tests/test_pipeline.py` | Unit tests (10 tests) |
| `tests/test_js_parity.py` | 5-case Python ↔ JS parity suite |
| `outputs/model_comparison.json` | Model benchmarks |
| `outputs/holdout_predictions.csv` | Holdout predictions & residuals |
| `outputs/plots/` | Static PNG charts |
| `data/processed/` | Train/holdout splits & engineered features |
