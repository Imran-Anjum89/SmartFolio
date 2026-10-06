# SmartFolio: An Adaptive Feedback-Driven Machine Learning Framework for Portfolio Optimization and Investment Decision Making

SmartFolio is an intelligent decision-support system integrating **XGBoost & PyTorch LSTM Return Forecasting**, **Ledoit-Wolf Covariance Shrinkage**, **Markowitz Portfolio Optimization**, **Leakage-Free Walk-Forward Backtesting (E0–E6)**, **Dual Feedback Tracking**, and a **Portfolio-Sharpe Champion–Challenger Adaptive Retraining Mechanism**.

Evaluated on the Indian equity market using the liquid **NIFTY 50 universe**, with explicit consideration of survivorship bias, Indian market transaction fees (15 bps), and realistic allocation constraints ($0 \le w_i \le 0.40$, $\sum w_i = 100\%$).

---

## 📌 Key Architectural Innovations

1. **Prediction Accuracy vs Portfolio Decision Quality (Gap 1)**:
   - Evaluates models on two distinct tracks: statistical forecasting metrics ($\text{MAE}, \text{RMSE}, R^2, \text{Directional Accuracy}$) and portfolio decision metrics ($\text{Sharpe Ratio}, \text{Sortino Ratio}, \text{Calmar Ratio}, \text{Max Drawdown}, \text{Turnover}$).

2. **Portfolio-Sharpe Champion–Challenger Retraining Gate (Gap 4)**:
   - Avoids blind daily retraining.
   - Retrains a Challenger candidate on accumulated historical + feedback observations.
   - Evaluates Champion vs Challenger on unseen out-of-sample data.
   - Promotes Challenger to New Champion **only if** $\text{Sharpe}_{\text{Challenger}} > \text{Sharpe}_{\text{Champion}}$ on out-of-sample portfolios, protecting the system against negative adaptation.

3. **Ledoit-Wolf Shrinkage Covariance (Gap 5)**:
   - Shrinks sample covariance toward constant correlation target, eliminating noise and singularity in financial covariance matrices.

4. **Leakage-Free Walk-Forward Evaluation with Indian Fees (Gap 3 & 6)**:
   - Tests experiments E0 through E6 strictly moving forward in time without future data leakage.
   - Incorporates realistic Indian transaction costs (STT + SEBI + Brokerage = 15 bps per trade).

5. **Allocation-Level Rule-Based Explainability (Gap 7)**:
   - Generates transparent, rule-based rationale for every asset weight recommendation (expected return driver, volatility factor, constraint ceiling).

---

## 🔬 Experimental Framework (E0 to E6)

| ID | Experiment Name | Model Pipeline | Covariance | Strategy / Features |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | Equal Weight | None ($w_i = 1/N$) | N/A | Fixed 1/N Benchmark |
| **E1** | Traditional Markowitz | Historical Mean Return | Sample Covariance | Standard Markowitz MPT |
| **E2** | Markowitz + Ledoit-Wolf | Historical Mean Return | Ledoit-Wolf Shrinkage | Covariance Noise Reduction |
| **E3** | XGBoost + Markowitz | XGBoost Regressor | Ledoit-Wolf Shrinkage | Tabular Gradient Boosting |
| **E4** | LSTM + Markowitz | PyTorch LSTM Regressor | Ledoit-Wolf Shrinkage | Sequential Deep Learning |
| **E5** | XGB-LSTM Ensemble | XGBoost + PyTorch LSTM | Ledoit-Wolf Shrinkage | Validation-Weighted Ensemble |
| **E6** | **SmartFolio Adaptive** | Champion ML Model | Ledoit-Wolf Shrinkage | **Full Feedback + Adaptive Retraining + Portfolio Sharpe Gate + Indian Fees** |

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.10+ is installed.

```bash
git clone https://github.com/YourRepo/SmartFolio.git
cd SmartFolio
pip install -r requirements.txt
```

### 2. Running Automated Tests

Run the complete test suite covering indicators, leakage checks, ML models, optimizer constraints, feedback gates, explainability, and walk-forward backtesting:

```bash
pytest -v
```

### 3. Launching the Web Application

Start the FastAPI backend with embedded glassmorphic frontend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📂 Project Structure

```
SmartFolio/
│
├── config/
│   ├── settings.py                     # Environment variables, RANDOM_SEED=42, RETRAIN_INTERVAL_DAYS=21
│   └── universe.json                   # Versioned Indian NIFTY 50 stock universe definition
│
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI Application entrypoint
│       │
│       ├── api/                        # REST API Endpoints
│       │   ├── stocks.py               # Stock universe & indicator endpoints
│       │   ├── predictions.py          # Forecasts & model comparison
│       │   ├── portfolios.py           # Portfolio optimization & decision engine endpoints
│       │   ├── feedback.py             # Dual feedback & adaptive retraining endpoints
│       │   └── backtesting.py          # Walk-forward backtest (E0-E6) endpoints
│       │
│       ├── services/                   # Business Logic & Core Engines
│       │   ├── data_service.py         # yfinance ingestion + synthetic fallback + data_source tagging
│       │   ├── feature_service.py      # RSI, MACD, SMA, EMA, Volatility, ROC calculation
│       │   ├── ml_service.py           # XGBoost, PyTorch LSTM, Ensemble & Champion-Challenger Selector
│       │   ├── feedback_service.py     # Dual prediction & portfolio feedback tracking
│       │   ├── risk_service.py         # Ledoit-Wolf Covariance & Risk metrics
│       │   ├── optimizer_service.py    # Markowitz Engine (Max Sharpe, Min Vol, Risk-Aware)
│       │   ├── explainer_service.py    # Allocation-level rule-based explainability
│       │   └── backtest_service.py     # Walk-forward Backtester with Indian fees & Bootstrap CIs
│       │
│       ├── models/                     # SQLAlchemy DB Models
│       │   └── db_models.py
│       ├── schemas/                    # Pydantic Schemas
│       │   └── schemas.py
│       └── database/                   # Database setup (SQLite default + PostgreSQL schema.sql)
│           ├── connection.py
│           └── schema.sql
│
├── ml/
│   ├── features/                       # Technical Feature Pipeline
│   │   ├── indicators.py
│   │   └── leakage_checks.py          # Automated temporal leakage verification
│   ├── models/                         # ML Models
│   │   ├── baseline.py                 # Historical Mean Baseline
│   │   ├── xgboost_model.py            # XGBoost Regressor
│   │   ├── lstm_model.py               # PyTorch LSTM Regressor
│   │   └── ensemble.py                 # Champion-Challenger & Weighted Ensemble
│   ├── feedback/                       # Feedback Loop Manager
│   │   └── feedback_loop.py
│   └── evaluation/                     # Metrics (MAE, RMSE, R², Sharpe, Sortino, Calmar, Bootstrap)
│       └── evaluate.py
│
├── frontend/                           # Glassmorphic Single-Page Web Dashboard
│   ├── index.html                      # Single-page app layout with tabbed UI
│   ├── css/
│   │   └── styles.css                  # Dark-theme layout & typography
│   └── js/
│       ├── app.js                      # UI routing & controller
│       ├── optimizer.js                # Portfolio setup, allocation & rule-based explanation UI
│       ├── ml_dashboard.js             # Model comparison (XGBoost vs LSTM vs Ensemble) UI
│       ├── feedback.js                 # Dual Feedback log & Champion-Challenger Retraining UI
│       ├── backtest.js                 # Walk-Forward Benchmark (E0-E6) charts & metrics
│       └── stock_explorer.js          # Interactive technical indicator viewer
│
└── tests/                              # Pytest Suite
    ├── test_data.py                    # Market data cleaning & data_source tagging tests
    ├── test_features.py                # Technical feature tests & leakage checks
    ├── test_ml.py                      # XGBoost, PyTorch LSTM, Ensemble tests
    ├── test_feedback.py                # Dual feedback & Portfolio-Sharpe promotion/rejection tests
    ├── test_optimizer.py               # Ledoit-Wolf & Markowitz constraint tests
    ├── test_explainer.py               # Rule-based explainability tests
    └── test_backtester.py              # Walk-forward strategy evaluation (E0-E6) tests
```
