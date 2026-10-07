# SmartFolio: Weeks 1–6 Quantitative ML Portfolio Decision System

**SmartFolio: Point-in-Time Indian Equity Analytics, ML Forecasting, and Risk-Aware Markowitz Portfolio Optimization**

> **Academic Milestone:** Phase 1 Deliverables (Weeks 1 to 6)  
> **Market:** Indian Equities, National Stock Exchange (NSE) NIFTY 50  
> **Research Foundations:** Point-in-time data architecture, leakage-free feature engineering, Ledoit-Wolf covariance shrinkage, and ML return forecasting (XGBoost & PyTorch LSTM)  
> **Core Principle:** Portfolio quality and statistical robustness matter more than raw prediction accuracy.

---

## 1. System Architecture (Weeks 1–6)

```text
Point-in-time market data (NIFTY 50 universe, historical OHLCV)
        ↓
Leakage & survivorship safeguards (Universe(t) point-in-time eligibility)
        ↓
Feature engineering (Price, Technical indicators, Volatility, Sector relative)
        ↓
Strict temporal train/val split (Zero look-ahead bias validation)
        ↓
ML Return Forecasting:
  ├── Historical Mean Baseline (Week 4 benchmark)
  ├── XGBoost Regressor (Week 5 tabular gradient boosting)
  └── PyTorch LSTM Regressor (Week 6 sequential recurrent neural network)
        ↓
Expected-return forecasts (μ̂)  +  Ledoit–Wolf covariance shrinkage (Σ̂_LW)
        ↓
Constrained Markowitz Portfolio Optimizer:
  [max μ̂ᵀw - (λ/2)wᵀΣ̂w  s.t.  ∑w_i = 1,  0 ≤ w_i ≤ 0.40]
        ↓
Allocation Decision Rationale & SHAP-based Explainability
        ↓
Glassmorphic Web Dashboard & FastAPI REST Service
```

---

## 2. Research Questions & Hypotheses (Weeks 1–6)

- **H1:** Machine learning-based return forecasts (XGBoost, LSTM) provide superior directional and risk-adjusted inputs for portfolio construction relative to simple historical mean extrapolation.
- **H2:** Ledoit–Wolf covariance shrinkage ($\Sigma_{\text{LW}}$) produces significantly better conditioned covariance matrices than empirical sample covariance, preventing extreme or unstable portfolio weights.
- **H3:** Sequential deep learning (PyTorch LSTM) captures temporal autocorrelations in market returns that complement tree-based gradient boosted models (XGBoost).

---

## 3. Experimental Ladder (E0 to E4)

| Experiment | Strategy Name | Return Forecast ($\hat{\mu}$) | Covariance Matrix ($\hat{\Sigma}$) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | Equal Weight | None ($w_i = 1/N$) | N/A | Naive $1/N$ Benchmark |
| **E1** | Classical Markowitz | Historical Mean Return | Sample Covariance | Classical 1952 Modern Portfolio Theory |
| **E2** | Markowitz + Ledoit-Wolf | Historical Mean Return | Ledoit-Wolf Shrinkage | Covariance Noise Reduction via Shrinkage |
| **E3** | XGBoost + Markowitz | XGBoost Regressor | Ledoit-Wolf Shrinkage | Tabular Non-linear Gradient Boosting |
| **E4** | PyTorch LSTM + Markowitz | PyTorch LSTM Regressor | Ledoit-Wolf Shrinkage | Sequential Deep Recurrent Network |

---

## 4. Weekly Milestone Breakdown

- **Week 1: Problem Definition & Research Foundations**  
  Formulated formal mathematical problem statement, research hypotheses $H_1–H_3$, and experimental ladder configurations (`configs/base.yaml`, `configs/experiments/e0.yaml` to `e4.yaml`).
- **Week 2: Point-in-Time Data Architecture & Ingestion**  
  Engineered point-in-time universe loader (`config/universe.json`), robust multi-symbol ingestion engine with real-world synthetic fallback (`backend/app/services/data_service.py`), and data validation suite.
- **Week 3: Feature Engineering & Leakage Protection**  
  Created 18+ technical and statistical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Rolling Volatility, Momentum) in `ml/features/indicators.py`, with strict temporal leakage assertions in `ml/features/leakage_checks.py`.
- **Week 4: Financial Baselines & Ledoit-Wolf Covariance**  
  Implemented Historical Mean Return baseline (`ml/models/baseline.py`), Ledoit-Wolf covariance shrinkage estimator (`backend/app/services/risk_service.py`), and constrained Markowitz Mean-Variance optimizer (`backend/app/services/optimizer_service.py`).
- **Week 5: ML Model A — XGBoost Regressor & Explainability**  
  Built gradient boosted decision tree forecaster (`ml/models/xgboost_model.py`) and SHAP-based feature attribution engine (`ml/explainability/xgb_shap.py`) with allocation decision rationales.
- **Week 6: ML Model B — PyTorch Sequential LSTM Network**  
  Developed sequential sliding-window PyTorch LSTM network (`ml/models/lstm_model.py`) for multi-step temporal pattern extraction and comparative model benchmarking.

---

## 5. Repository Structure

```text
SmartFolio/
│
├── configs/                            # Configuration files
│   ├── base.yaml                       # Base configuration (Seed 42, lookbacks)
│   ├── costs.yaml                      # Indian market statutory levies & fees (15 bps)
│   └── experiments/                    # Experiment definitions (E0 to E4)
│       ├── e0.yaml
│       ├── e1.yaml
│       ├── e2.yaml
│       ├── e3.yaml
│       └── e4.yaml
│
├── data/                               # Data architecture
│   ├── raw/                            # Ingested OHLCV datasets
│   ├── processed/                      # Feature matrices & adjusted prices
│   ├── research/                       # Predictions & portfolio records
│   ├── metadata/                       # Point-in-time universe records
│   └── cache/                          # Cache directory
│
├── ml/                                 # Machine Learning & Quantitative Finance Core
│   ├── features/
│   │   ├── indicators.py               # Technical features without look-ahead bias
│   │   └── leakage_checks.py           # Mathematical temporal leakage assertions
│   ├── models/
│   │   ├── baseline.py                 # Historical mean benchmark
│   │   ├── xgboost_model.py            # XGBoost Regressor
│   │   └── lstm_model.py               # PyTorch LSTM Regressor
│   ├── explainability/
│   │   └── xgb_shap.py                 # Feature attribution & decision rationale
│   └── evaluation/
│       └── evaluate.py                 # Sharpe, Sortino, Calmar, CVaR 95%
│
├── backend/                            # FastAPI REST API Backend
│   └── app/
│       ├── main.py                     # Entrypoint & static UI mount
│       ├── api/                        # REST routes (stocks, predictions, portfolios)
│       ├── services/                   # Engines (Data, Risk, Optimizer, Explainer, ML)
│       ├── models/db_models.py         # SQLAlchemy ORM models
│       ├── schemas/schemas.py          # Pydantic schemas
│       └── database/schema.sql         # SQL DDL (PostgreSQL & SQLite)
│
├── frontend/                           # Glassmorphic Interactive Web Dashboard
│   ├── index.html                      # Single-page interface
│   ├── css/styles.css                  # Modern dark glassmorphic design system
│   └── js/
│       ├── app.js                      # Router & controller
│       ├── optimizer.js                # Portfolio setup & explainability modal
│       ├── ml_dashboard.js             # Model comparison & feature attribution
│       ├── stock_explorer.js           # Technical indicator candlestick explorer
│       └── compliance.js               # Accessibility & consent manager
│
├── tests/                              # Automated Pytest Suite (12 / 12 passing)
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_ml.py
│   ├── test_optimizer.py
│   └── test_explainer.py
│
├── run_experiment.py                   # CLI: Run individual experiments (E0 to E4)
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## 6. Verification & Automated Test Suite (12 / 12 Passed)

Run the automated test suite verifying all Weeks 1 to 6 components:

```powershell
python -m pytest -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.2, pytest-9.1.1
rootdir: C:\Users\mdimr\OneDrive\Desktop\SmartFolio
collected 12 items

tests/test_data.py::test_universe_loading PASSED                         [  8%]
tests/test_data.py::test_fetch_stock_data_source_tagging PASSED          [ 16%]
tests/test_explainer.py::test_allocation_explainability PASSED           [ 25%]
tests/test_features.py::test_compute_technical_indicators PASSED         [ 33%]
tests/test_features.py::test_generate_target PASSED                      [ 41%]
tests/test_features.py::test_leakage_checks PASSED                       [ 50%]
tests/test_ml.py::test_historical_mean_baseline PASSED                   [ 58%]
tests/test_ml.py::test_xgboost_model PASSED                              [ 66%]
tests/test_ml.py::test_lstm_model PASSED                                 [ 75%]
tests/test_optimizer.py::test_ledoit_wolf_covariance PASSED              [ 83%]
tests/test_optimizer.py::test_markowitz_constraints PASSED               [ 91%]
tests/test_optimizer.py::test_risk_profiles_behavior PASSED              [100%]

============================= 12 passed in 11.23s =============================
```

---

## 7. How to Run

### 7.1 Web Dashboard & API Server (Unified on Port 8000)
```powershell
python -m uvicorn backend.app.main:app --port 8000 --reload
```
- **Web Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **REST API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 7.2 CLI Experiment Runner
```powershell
# Run individual experiments from E0 to E4
python run_experiment.py --experiment E0
python run_experiment.py --experiment E1
python run_experiment.py --experiment E2
python run_experiment.py --experiment E3
python run_experiment.py --experiment E4
```
