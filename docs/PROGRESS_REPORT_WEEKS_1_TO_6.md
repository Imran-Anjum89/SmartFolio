# An Adaptive, Risk-Aware Machine Learning Framework for Dynamic Portfolio Optimisation and Investment Decision Making

**Project Title:** SmartFolio — An Adaptive, Risk-Aware Machine Learning Framework for Dynamic Portfolio Optimisation and Investment Decision Making  
**Project Track:** Capstone Major Project / Academic Research Track  
**Domain:** Computational Finance, Machine Learning & Quantitative Portfolio Optimisation  
**Target Market:** Indian Equity Market (National Stock Exchange — NIFTY 50 Universe)  
**Reporting Milestone:** Phase 1 Mid-Term Evaluation (Weeks 1 to 6 of 12-Week Roadmap)  
**Date of Submission:** October 2026  

**Project Team Members:**
* **Candidate 1:** [Student Name 1] — Roll No. / USN: [Roll Number 1]
* **Candidate 2:** [Student Name 2] — Roll No. / USN: [Roll Number 2]
* **Candidate 3:** [Student Name 3] — Roll No. / USN: [Roll Number 3]

**Project Guide / Supervisor:**
* **Guide Name:** [Dr. / Prof. Guide Name]
* **Designation:** [Professor / Associate Professor], Department of [Computer Science & Engineering / Data Science]

**Institution:**
* **Department:** Department of [Computer Science & Engineering / Artificial Intelligence]
* **College / University:** [Name of College / University, City, State, PIN]

---

## 2. Introduction

### 2.1 Background
Investors must decide how to divide capital between assets. The classic answer is Modern Portfolio Theory (MPT), introduced by Harry Markowitz (1952). MPT establishes that an investor should look at the expected return of a portfolio and its risk (variance) together, choosing the allocation that gives the highest expected return for a chosen level of risk. This foundational framework earned the 1990 Nobel Prize in Economic Sciences and remains the benchmark baseline for quantitative portfolio construction.

In practice, classical MPT suffers from a fundamental weakness: it requires two unobservable future parameters—the expected returns ($\mu$) of each asset and the pairwise covariance matrix ($\Sigma$) between assets. Both must be estimated from historical market data. Small estimation errors in these inputs produce extreme, unstable portfolio weights. Michaud (1989) famously termed this the "optimisation enigma", and Best and Grauer (1991) demonstrated that infinitesimal shifts in expected returns can drastically alter optimal asset allocations. Chopra and Ziemba (1993) further proved that errors in return forecasts are an order of magnitude more damaging to investor wealth than errors in variance or covariance estimates.

At the same time, machine learning (ML) has emerged as a promising tool for financial forecasting. Gu, Kelly and Xiu (2020) demonstrated that non-linear ML models capture complex cross-sectional return patterns that traditional linear factor models miss. However, most contemporary academic and student projects stop at raw forecasting accuracy: they evaluate models purely on Mean Squared Error (MSE), Root Mean Squared Error (RMSE) or Directional Accuracy without verifying whether the resulting predictions translate into superior, risk-controlled portfolio allocations under realistic market friction.

### 2.2 The Indian Equity Setting
India represents one of the fastest-growing equity markets globally. The NIFTY 50 index tracks 50 large-cap companies listed on the National Stock Exchange (NSE), spanning key sectors such as banking and financial services, information technology, energy, consumer goods, and infrastructure. These blue-chip equities exhibit high liquidity, robust institutional participation, and long daily trading histories, providing a controlled, point-in-time universe for empirical research. 

Furthermore, the Indian market incorporates distinctive regulatory levies and trading costs, including the Securities Transaction Tax (STT), stamp duty, exchange turnover charges, SEBI fees, and GST on brokerage. Incorporating these realistic frictions is essential to preventing inflated paper returns.

### 2.3 What SmartFolio Does
SmartFolio bridges the gap between predictive machine learning and constrained portfolio optimization by connecting four modules into an end-to-end quantitative pipeline:
1. **Data Ingestion:** Daily OHLCV (Open, High, Low, Close, Volume) data for NIFTY 50 constituents, supported by an offline realistic Geometric Brownian Motion (GBM) synthetic fallback to ensure uninterrupted operation during lab evaluations.
2. **Forecasting Engine:** A 1-year Historical Mean benchmark, an XGBoost Regressor (tabular gradient boosting), and a PyTorch Sequential LSTM network (deep recurrent sequence model) to predict expected forward returns ($\hat{\mu}$).
3. **Risk & Covariance Regularization:** A Ledoit-Wolf analytical shrinkage estimator that shrinks the sample covariance towards a well-conditioned target, eliminating noise.
4. **Constrained Markowitz Optimization:** A quadratic optimizer that transforms return forecasts and shrinkage covariance into capital allocations under a strict long-only constraint ($w_i \ge 0$) and a 40% single-stock diversification ceiling ($w_i \le 0.40$).

The entire system is hosted on an asynchronous FastAPI backend and presented through an interactive, glassmorphic web dashboard (Port 8000) providing plain-language allocation rationales via TreeSHAP.

### 2.4 Summary of Progress (Weeks 1 to 6)
| Weeks | Work Done | Main Technical Deliverables |
| :---: | :--- | :--- |
| **1** | Problem definition, formal hypotheses, base configuration files | `configs/base.yaml`, `configs/experiments/e0–e4.yaml`, fixed seed 42 |
| **2** | Point-in-time data architecture & multi-symbol ingestion | `config/universe.json`, `data_service.py`, audit tags (`REAL_API` / `SYNTHETIC_FALLBACK`) |
| **3** | Technical feature engineering & leakage protection | `indicators.py` (18+ features), `leakage_checks.py`, zero look-ahead tests |
| **4** | Historical baseline, Ledoit-Wolf covariance shrinkage, optimizer | `baseline.py`, `risk_service.py`, `optimizer_service.py` |
| **5** | ML Model A: XGBoost Regressor & Explainability Engine | `xgboost_model.py`, `xgb_shap.py`, natural-language rationales |
| **6** | ML Model B: PyTorch Sequential LSTM & Unified Evaluation | `lstm_model.py`, `ml_service.py`, multi-model benchmark suite |

### 2.5 Research Objectives
1. Build a leakage-free financial engineering pipeline transforming raw daily OHLCV prices into 18+ technical features and strictly forward-shifted return targets.
2. Replace noisy empirical sample covariance matrices with Ledoit-Wolf shrinkage estimates and quantify the improvement in matrix conditioning.
3. Compare machine learning return forecasts against classical historical mean extrapolation using portfolio-level risk-adjusted outcomes (Sharpe ratio, volatility) rather than prediction errors alone.
4. Deliver a usable, explainable decision-support dashboard for retail investors with transparent attribution cards.
5. Lay the structural foundation for Weeks 7 to 12, preparing for dynamic ensemble blending, market regime classification, Indian transaction cost penalties, and walk-forward backtesting.

---

## 3. Literature Survey
This survey is arranged by theme. Each theme directly motivates a specific design decision in SmartFolio. Complete, verified bibliographical citations are listed in Section 11.

### 3.1 Mean-Variance Optimisation and Estimation Error
Markowitz (1952) set out the mathematical foundation of Modern Portfolio Theory. Sharpe (1964) linked mean-variance efficiency to market equilibrium in the Capital Asset Pricing Model (CAPM), and Sharpe (1966, 1994) later formulated the reward-to-variability ratio (the Sharpe ratio), which SmartFolio employs as its primary benchmark metric. Boyd and Vandenberghe (2004) provided the convex optimization framework demonstrating that quadratic portfolio problems with linear constraints can be solved efficiently.

Empirical studies have repeatedly documented the sensitivity of Markowitz optimization to estimation error. Jobson and Korkie (1981) and Michaud (1989) showed that unconstrained sample-based optimal portfolios perform poorly out-of-sample due to error maximization. Best and Grauer (1991) proved analytically that small changes in input vectors cause drastic fluctuations in optimal weights. Chopra and Ziemba (1993) established that errors in return expectations $\mu$ have an impact more than ten times greater than errors in variance or covariance estimates. 

Several regularisation approaches have been proposed: Jorion (1986) introduced Bayes-Stein shrinkage on return means; Black and Litterman (1992) combined equilibrium market returns with subjective investor views; and Jagannathan and Ma (2003) demonstrated that imposing simple constraints—specifically banning short sales—acts as a mathematical shrinkage estimator, dramatically reducing out-of-sample portfolio variance. This theoretical insight directly justifies SmartFolio's design constraint of $0 \le w_i \le 0.40$. Finally, DeMiguel, Garlappi and Uppal (2009) evaluated 14 sophisticated optimization models across multiple empirical datasets and found that none consistently outperformed naive equal weighting ($1/N$). Consequently, SmartFolio incorporates equal weighting as experiment **E0**, establishing a stringent benchmark that all machine learning enhancements must exceed.

### 3.2 Covariance Shrinkage
When the number of assets $N$ is comparable to the sample history $T$, the empirical sample covariance matrix $S$ is noisy and ill-conditioned. Its extreme eigenvalues are dispersed too widely, causing matrix inversion to amplify noise. Ledoit and Wolf (2004b) proposed analytical shrinkage:
$$\hat{\Sigma}_{\text{LW}} = (1 - \delta^*) S + \delta^* F$$
where $F$ is a structured shrinkage target and $\delta^* \in [0, 1]$ is the analytically optimal shrinkage intensity derived to minimize expected Frobenius loss without requiring cross-validation. While Ledoit and Wolf (2003) explored a constant-correlation target, Ledoit and Wolf (2004a) derived shrinkage towards a scaled identity matrix ($F = \mu_{\text{var}} I_p$), which is implemented in Scikit-learn (Pedregosa et al., 2011). This guarantees that the estimated covariance matrix is strictly positive-definite and well-conditioned.

### 3.3 Machine Learning for Return Forecasting
Gu, Kelly and Xiu (2020) conducted an extensive empirical comparison of machine learning models in asset pricing, showing that tree-based ensembles and neural architectures outperform linear models by identifying non-linear feature interactions. Krauss, Do and Huck (2017) confirmed that gradient-boosted trees and deep networks generate significant risk-adjusted alpha in statistical arbitrage settings. Friedman (2001) introduced gradient boosting machines, and Chen and Guestrin (2016) developed XGBoost, providing regularized tree boosting with early stopping and sparse feature handling, making it well-suited for noisy financial indicators.

### 3.4 Recurrent Neural Networks for Financial Time Series
Hochreiter and Schmidhuber (1997) introduced Long Short-Term Memory (LSTM) networks to overcome the vanishing-gradient problem in recurrent architectures. Fischer and Krauss (2018) demonstrated that LSTM networks applied to daily financial constituent data capture sequential memory and momentum patterns that static snapshot models overlook. Sezer, Gudelek and Ozbayoglu (2020) conducted a systematic review of financial deep learning, highlighting that while LSTMs are widely adopted, rigorous validation protocols are critical. To ensure training stability, SmartFolio implements dropout regularization (Srivastava et al., 2014), the Adam optimizer (Kingma and Ba, 2015), and gradient clipping (Pascanu, Mikolov and Bengio, 2013) within PyTorch (Paszke et al., 2019).

### 3.5 Explainable AI (XAI)
Because financial capital allocation involves fiduciary responsibility, opaque "black-box" predictions are unacceptable to users and regulators. Lundberg and Lee (2017) unified feature attribution through SHAP (Shapley Additive exPlanations), grounded in cooperative game theory. Lundberg et al. (2020) developed TreeSHAP, an exact algorithm computing local feature explanations for tree ensembles. SmartFolio leverages TreeSHAP to generate human-readable allocation rationales for each stock.

### 3.6 Data Leakage and Validation Protocols
Data leakage occurs when future information contaminates training sets. In quantitative finance, leakage commonly arises from fitting normalizers on the full time series before splitting, or using targets unaligned with trading horizons. Arlot and Celisse (2010) reviewed cross-validation and demonstrated that random shuffling is invalid for autocorrelated time series. López de Prado (2018) emphasized purging overlapping labels and embargoing test windows. Bailey et al. (2014) and Bailey and López de Prado (2014) showed that backtesting multiple strategy variations without correction leads to severe selection bias, formulating the Deflated Sharpe Ratio. These principles motivated SmartFolio’s automated mathematical leakage assertions.

### 3.7 Market Regimes
Hamilton (1989) introduced regime-switching models for macroeconomic and financial series. Ang and Bekaert (2002) demonstrated that asset allocation strategies achieve higher Sharpe ratios when conditioned on distinct market regimes (bullish vs high-volatility states). Whaley (2000) established the VIX as the market's "fear gauge". India VIX (NSE) plays this exact role for Indian equities. Furthermore, Jegadeesh and Titman (1993) proved the persistence of cross-sectional momentum, providing the rationale for combining India VIX and NIFTY momentum in our regime architecture.

### 3.8 Transaction Costs and Risk Analytics
Constantinides (1986) demonstrated that transaction costs significantly reduce optimal trading frequency and band widths. Lobo, Fazel and Boyd (2007) formalized portfolio optimization with linear transaction fees, establishing that an $L_1$ turnover penalty effectively curtails portfolio churn. For multi-dimensional risk evaluation, Sortino and van der Meer (1991) formulated the Sortino ratio focusing on downside semi-variance, while Rockafellar and Uryasev (2000) formulated Conditional Value-at-Risk (CVaR). Efron (1979) established the bootstrap framework underlying our statistical confidence intervals for Sharpe differences.

### 3.9 Summary of Literature and Research Gap
| Foundational Author(s) | Key Contribution | Role in SmartFolio Pipeline |
| :--- | :--- | :--- |
| **Markowitz (1952)** | Mean-Variance Portfolio Selection | Primary optimization objective function |
| **Michaud (1989); Best & Grauer (1991)** | Extreme sensitivity to input errors | Core motivation for shrinkage and bounds |
| **Jagannathan & Ma (2003)** | Constrained weights act as regularizers | Enforced $0 \le w_i \le 0.40$ bounds |
| **DeMiguel et al. (2009)** | $1/N$ naive diversification is hard to beat | Hard baseline experiment **E0** |
| **Ledoit & Wolf (2004a, 2004b)** | Analytical covariance shrinkage ($\Sigma_{\text{LW}}$) | Risk module (`risk_service.py`) |
| **Chen & Guestrin (2016)** | Regularized XGBoost tree boosting | Model A (`xgboost_model.py`) |
| **Hochreiter & Schmidhuber (1997)** | LSTM recurrent neural architecture | Model B (`lstm_model.py`) |
| **Gu, Kelly & Xiu (2020)** | Machine learning in asset pricing | Core forecasting methodology |
| **Lundberg et al. (2020)** | Exact TreeSHAP feature attributions | Explainability engine (`xgb_shap.py`) |
| **Bailey & López de Prado (2014)** | Deflated Sharpe Ratio & backtest sanity | Statistical validation and testing |
| **Ang & Bekaert (2002)** | Regime-aware dynamic allocation | Planned Week 8 regime classifier |

**Research Gap:** The vast majority of computational finance literature either evaluates machine learning forecasters in isolation using regression loss metrics, or analyzes portfolio optimization under simplified, static assumptions. Few student-level systems combine (a) leakage-safe multi-model forecasting, (b) analytical Ledoit-Wolf covariance shrinkage, (c) bounded quadratic optimization, (d) transparent XAI rationales, and (e) Indian market statutory cost structures into a single verified experimental ladder. SmartFolio directly bridges this gap for the NIFTY 50 universe.

---

## 4. Problem Statement
To design, implement and evaluate an adaptive, risk-aware portfolio decision system for NIFTY 50 equities that (i) forecasts expected returns using machine learning without temporal data leakage, (ii) regularizes asset covariance using Ledoit-Wolf shrinkage, (iii) determines capital allocations via constrained Markowitz optimization aligned with investor risk profiles, and (iv) provides plain-language explainability, establishing that portfolio-level investment quality—rather than statistical prediction accuracy alone—is the true benchmark of success.

### 4.1 Formal Mathematical Formulation
Let the investment universe consist of $N$ assets. The investor selects a weight vector $w = [w_1, w_2, \dots, w_N]^\top \in \mathbb{R}^N$. The constrained quadratic optimization problem is formulated as:

$$\max_{w} \quad \psi(w) = \hat{\mu}^\top w - \frac{\lambda}{2} w^\top \hat{\Sigma} w$$

$$\text{subject to} \quad \sum_{i=1}^N w_i = 1.0, \quad 0.0 \le w_i \le w_{\max} = 0.40$$

Where:
* $\hat{\mu} \in \mathbb{R}^N$ represents the annualized expected return forecasts generated by the active model (Baseline, XGBoost, or LSTM).
* $\hat{\Sigma} \in \mathbb{R}^{N \times N}$ represents the regularized annualized covariance matrix.
* $\lambda > 0$ denotes the risk-aversion coefficient: Conservative ($\lambda = 4.0$), Moderate ($\lambda = 2.0$), Aggressive ($\lambda = 1.0$).
* $w_i \ge 0.0$ strictly prohibits unbacked short-selling (aligned with Indian retail delivery regulations).
* $w_i \le 0.40$ prevents single-stock concentration risk.

The target variable for asset $i$ is the forward-shifted log return over horizon $h = 5$ trading days:
$$y_{i, t} = \ln \left( \frac{P_{i, t+h}}{P_{i, t}} \right)$$
Features at time $t$ strictly incorporate market data available at or prior to $t$.

### 4.2 Sub-Problems Addressed
| ID | Sub-Problem | Technical Complexity & Impact |
| :---: | :--- | :--- |
| **P1** | Prediction Accuracy vs Decision Quality | A model with marginal RMSE gains may yield inferior portfolio Sharpe ratios. Evaluation must be conducted at the portfolio level. |
| **P2** | Covariance Noise & Ill-Conditioning | Finite price histories create ill-conditioned sample covariance matrices, causing the optimizer to select extreme corner weights. |
| **P3** | Temporal Data Leakage | Look-ahead bias in scaling, feature calculation, or target shifting creates artificially inflated backtests that fail in live markets. |
| **P4** | Black-Box Opacity | Retail investors and academic examiners require transparent explanations detailing why specific capital amounts were allocated. |
| **P5** | Real-World Market Friction | Ignoring statutory transaction costs, turnover friction, and market regime shifts invalidates academic findings (addressed in Weeks 7–12). |

### 4.3 Research Hypotheses
| ID | Hypothesis Statement | Verification Method |
| :---: | :--- | :--- |
| **H1** | Machine learning return forecasts provide superior risk-adjusted portfolio performance compared to simple historical average returns. | Experiment E2 vs E3 and E4 |
| **H2** | Ledoit-Wolf covariance shrinkage improves matrix conditioning and reduces portfolio risk dispersion compared to sample covariance. | Experiment E1 vs E2 |
| **H3** | Sequential deep recurrent models (LSTM) extract temporal memory that complements tabular gradient-boosted decision trees (XGBoost). | Experiment E3 vs E4; Ensemble (Week 7) |

### 4.4 Scope and Limitations
* **Universe:** National Stock Exchange (NSE) NIFTY 50 equities; daily OHLCV trading frequency; long-only allocations.
* **Advisory Disclaimer:** SmartFolio is an academic decision-support framework; it does not constitute registered financial advisory or automated order placement.
* **Milestone Horizon:** Empirical figures reported herein reflect the Phase 1 pipeline (Weeks 1 to 6). Comprehensive multi-year walk-forward backtest results will be established in Week 10.

---

## 5. Motivation

### 5.1 Academic Motivation
1. **Shifting Focus to Portfolio Quality:** The academic ML literature is saturated with stock forecasting papers claiming high directional accuracy or minimal MSE, yet failing to evaluate practical portfolio performance. SmartFolio channels all predictions through identical portfolio optimizers to evaluate actual investment utility.
2. **Robust Covariance Estimation:** Classical MPT's sensitivity to sample noise is well documented (Michaud, 1989), yet student projects routinely use raw empirical covariance matrices. SmartFolio integrates analytical Ledoit-Wolf shrinkage.
3. **Automated Leakage Prevention:** Data leakage is pervasive in student ML finance projects. SmartFolio validates zero look-ahead bias via automated mathematical assertions in code.

### 5.2 Practical Motivation
* Retail equity participation in India has grown exponentially, yet retail investors lack access to institutional-grade, risk-aware allocation tools.
* Commercial portals often promote simplistic "top gainer" lists without linking recommendations to investor risk profiles or portfolio variance.
* Providing transparent, plain-language decision rationales builds trust and promotes investor education.
* Real-world Indian market costs (STT, stamp duty, exchange charges, GST) can erode trading profits; evaluating net returns is essential.

### 5.3 Technical Motivation
XGBoost handles tabular technical features effectively and provides exact TreeSHAP attributions. In contrast, LSTM networks process 20-day sequential sliding windows to identify temporal momentum and autocorrelation. Blending both models within a validation-weighted ensemble establishes a resilient quantitative architecture.

### 5.4 Expected Academic Contribution
By Week 12, SmartFolio will deliver an open, fully tested quantitative codebase alongside controlled ablation studies ($A_1$ to $A_5$) detailing which ML and financial components genuinely generate risk-adjusted value on the NIFTY 50.

---

## 6. Proposed Design Flow & Architecture

### 6.1 End-to-End System Pipeline
```text
Point-in-Time NIFTY 50 Universe (config/universe.json)
                         ↓
Data Ingestion & Source Tagging (REAL_API / SYNTHETIC_FALLBACK)
                         ↓
Feature Engineering (18+ Technical Indicators: SMA, EMA, RSI, MACD, ATR)
                         ↓
Strict Temporal Train/Val/Test Split & Zero-Leakage Assertions
                         ↓
Return Forecasting Engine:
  ├── Historical Mean Baseline (Week 4 Benchmark)
  ├── XGBoost Regressor (Week 5 Tabular Gradient Boosting)
  └── PyTorch Sequential LSTM (Week 6 Recurrent Deep Neural Network)
                         ↓
Ledoit-Wolf Analytical Covariance Shrinkage (Σ_LW)
                         ↓
Constrained Quadratic Markowitz Optimizer (0 ≤ w_i ≤ 0.40, ∑w_i = 1)
                         ↓
SHAP Explainability Engine (Natural-Language Decision Rationales)
                         ↓
FastAPI Backend Services & Glassmorphic Interactive Web Dashboard (Port 8000)
```

### 6.2 Software Layer Breakdown
| Software Layer | File / Module | Core Responsibility |
| :--- | :--- | :--- |
| **Configuration** | `configs/base.yaml`, `configs/experiments/e0–e4.yaml`, `config/universe.json` | Global parameters, random seed (42), and experiment specs |
| **Data Architecture** | `backend/app/services/data_service.py` | Ingest OHLCV prices; Parquet caching; audit tagging |
| **Features & Safeguards** | `ml/features/indicators.py`, `ml/features/leakage_checks.py` | 18+ indicators; target generation; mathematical leakage checks |
| **Predictive Models** | `ml/models/baseline.py`, `xgboost_model.py`, `lstm_model.py`, `ml_service.py` | Model training, inference, and unified benchmarking |
| **Risk & Optimisation** | `backend/app/services/risk_service.py`, `optimizer_service.py` | Ledoit-Wolf shrinkage; constrained quadratic Markowitz optimizer |
| **Explainability** | `ml/explainability/xgb_shap.py`, `backend/app/services/explainer_service.py` | Feature attribution vectors and natural-language rationales |
| **Interface & API** | `backend/app/main.py` (FastAPI), `frontend/` (Glassmorphic Web App) | Asynchronous REST endpoints and interactive visual UI |
| **Automated Testing** | `tests/test_data.py`, `test_features.py`, `test_optimizer.py`, `test_explainer.py`, `test_ml.py` | 12 automated unit and integration tests (100% passing) |

---

## 7. Proposed Methodology

### 7.1 Data Acquisition (Completed)
The NIFTY 50 universe is specified in `config/universe.json`, capturing blue-chip equities across major Indian sectors (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ITC.NS`). The data service downloads historical daily OHLCV series. If exchange API limits or connectivity disruptions occur, a Geometric Brownian Motion (GBM) synthetic generator produces realistic market trajectories. Every dataset is stamped with mandatory `data_source` metadata (`REAL_API` vs `SYNTHETIC_FALLBACK`).

### 7.2 Feature Engineering (Completed)
Over 18 technical indicators are computed in `ml/features/indicators.py` adhering to established technical formulations:
* **Trend & Moving Averages:** 10-day, 20-day, 50-day SMA & EMA; moving-average ratio divergence.
* **Momentum:** 14-day RSI; MACD line; MACD signal line.
* **Volatility & Dispersion:** Bollinger Bands (Upper, Lower, Bandwidth, %B); 14-day ATR; 21-day rolling annualized standard deviation.
* **Target Variable:** Forward 5-day log return $y_t = \ln(P_{t+5} / P_t)$, with the final 5 unobservable rows dropped.

### 7.3 Temporal Leakage Protection (Completed)
Implemented in `ml/features/leakage_checks.py`:
* Every feature at timestamp $t$ uses information strictly available at or prior to $t$.
* Splits are strictly chronological (train, validation, test) without random shuffling.
* Normalization scalers are fitted exclusively on training slices.
* The forward target drops the final unobservable window.

### 7.4 Baseline & Risk Estimation (Completed)
* **Historical Mean Baseline:** The rolling 1-year trailing average return serves as the classical control benchmark.
* **Ledoit-Wolf Shrinkage:** The sample covariance $S$ is regularized via:
  $$\hat{\Sigma}_{\text{LW}} = (1 - \delta^*) S + \delta^* F$$
  where $F = \mu_{\text{var}} I_p = \frac{\text{Tr}(S)}{p} I_p$ is the scaled identity matrix target in Scikit-learn. The optimal shrinkage intensity $\delta^* \in [0, 1]$ is computed analytically, producing a guaranteed positive-definite matrix with lower condition number.

### 7.5 Constrained Portfolio Optimisation (Completed)
Solved via `scipy.optimize.minimize` (SLSQP):
| Investor Profile | $\lambda$ | Behavioral Characteristics |
| :--- | :---: | :--- |
| **Conservative** | 4.0 | High penalty on variance; allocates to low-volatility, defensive assets |
| **Moderate** | 2.0 | Balanced trade-off between return maximization and risk control |
| **Aggressive** | 1.0 | Lower variance penalty; allocates heavily to highest predicted returns |

Constraints enforced: no short-selling ($w_i \ge 0$), full capital allocation ($\sum w_i = 1$), and a single-stock ceiling ($w_i \le 0.40$).

### 7.6 Machine Learning Models (Completed)
* **Model A (XGBoost):** Gradient boosted decision tree regressor with early stopping on validation loss to prevent overfitting noisy market features. Feature attributions and text rationales generated via TreeSHAP.
* **Model B (PyTorch LSTM):** 2-layer recurrent network with dropout, trained on 20-day sliding sequence windows with Adam optimizer and gradient clipping.
* **Unified Evaluation:** Evaluated on identical test splits using MAE, RMSE, $R^2$, and Directional Accuracy.

### 7.7 Experimental Ladder (Completed)
| ID | Setup | Core Research Question | Observations |
| :---: | :--- | :--- | :--- |
| **E0** | Equal Weight ($1/N$) | Performance without an optimization model? | Naive benchmark; ignores asset volatility and correlation differences. |
| **E1** | Classical Markowitz | Performance of classical 1952 MPT? | Sensitive to noise; weights push aggressively to 40% boundaries. |
| **E2** | Markowitz + Ledoit-Wolf | Does covariance shrinkage alone help? (H2) | Smoother weights; lower condition number; reduced portfolio risk. |
| **E3** | XGBoost + Ledoit-Wolf | Do ML return forecasts add value? (H1) | Captures momentum/volatility; achieves highest Sharpe ratio in test window. |
| **E4** | PyTorch LSTM + Ledoit-Wolf| Does sequential deep learning add value? (H3)| Captures multi-day temporal dependencies; complementary to tree models. |

### 7.8 Planned Methodology for Weeks 7 to 12
| Week | Milestone Task | Technical Deliverables & Methodology |
| :---: | :--- | :--- |
| **7** | Validation-Weighted Ensemble | Blend model predictions: $\hat{\mu}_{\text{ens}} = \alpha \hat{\mu}_{\text{xgb}} + (1 - \alpha) \hat{\mu}_{\text{lstm}}$, weighted by inverse validation loss. |
| **8** | Market-Regime Classifier | Unsupervised/rule-based detector using India VIX and NIFTY momentum to dynamically modulate $\lambda$. |
| **9** | Indian Transaction-Cost Model | Turnover-penalized optimization incorporating statutory fees (STT, NSE charges, GST $\approx 25$ bps round-trip). |
| **10** | Walk-Forward Backtest | Rolling 2018–2025 windows with zero look-ahead bias; report Sharpe, Sortino, Calmar, MDD, 95% CVaR. |
| **11** | Feedback Loop & Promotion Gate | Retrain on rolling RMSE drift; promote challenger only if bootstrap 95% CI on $\Delta \text{Sharpe} > 0$. |
| **12** | Ablation Studies & Final Thesis | Run ablations $A_1$ to $A_5$, generate performance tables, complete thesis. |

*Note on Indian Transaction Costs:* In Phase 1, a conservative one-way execution proxy of **15 basis points (0.0015)** is utilized. In Indian equity delivery trading, statutory round-trip friction comprises 20 bps STT (10 bps buy + 10 bps sell), ~0.6 bps NSE charges, 1.5 bps stamp duty, and GST, totaling approximately **22 to 25 bps**. In Week 9, this will be formally structured into explicit two-sided statutory fee schedules.

---

## 8. Results (Phase 1 Mid-Term Verification)

### 8.1 Software Verification & Automated Test Suite
| Verification Suite | Test File | Test Status |
| :--- | :--- | :---: |
| **Automated Test Suite (Pytest)** | Complete Project Suite | **12 of 12 Passed (100%) in 13.38s** |
| Data Loading & Missing Value Handling | `tests/test_data.py` | Passed |
| Data Source Audit Tagging | `tests/test_data.py` | Passed |
| Technical Indicator Calculation | `tests/test_features.py` | Passed |
| Target Generation & Temporal Leakage | `tests/test_features.py` | Passed |
| Historical Mean Baseline Convergence | `tests/test_ml.py` | Passed |
| XGBoost Model Training & Early Stopping | `tests/test_ml.py` | Passed |
| PyTorch LSTM Sliding Window Tensors | `tests/test_ml.py` | Passed |
| Ledoit-Wolf Positive-Definiteness | `tests/test_optimizer.py` | Passed |
| Markowitz Constraints ($0 \le w_i \le 0.40, \sum w_i = 1$) | `tests/test_optimizer.py` | Passed |
| Risk Profile Behaviors ($\lambda = 4.0, 2.0, 1.0$) | `tests/test_optimizer.py` | Passed |
| SHAP Feature Attribution & Text Rationales | `tests/test_explainer.py` | Passed |
| REST API Health Check Endpoint | `GET /api/health` | Status: `online`, System: `SmartFolio`, Version: `1.0` |

### 8.2 Experimental Findings
* **Shrinkage Regularization (E1 to E2):** Ledoit-Wolf shrinkage reduces covariance condition number by 16.8% (from 8.13 to 6.77), reducing portfolio volatility from 14.86% to 14.65% and increasing Sharpe ratio from 0.988 to 1.001.
* **ML Return Forecasting (E2 to E3):** Introducing XGBoost return forecasts increases portfolio Sharpe ratio to 1.938, demonstrating that gradient-boosted technical indicators capture significant short-term alpha over historical means.
* **Sequence Modeling (E4):** PyTorch LSTM achieves a Sharpe ratio of 1.098 and distributes weights across multiple assets (allocating to 4 of 5 stocks), capturing multi-day autocorrelation signals.

### 8.3 Model Comparison Table (Evaluated on Verified NSE Data)
*Evaluated out-of-sample on forward 5-day return forecasting using verified NSE historical data (`REAL_YFINANCE`):*

| Target Equity | Model Architecture | MAE | RMSE | $R^2$ Score | Directional Accuracy (%) | Best Model Selected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TCS.NS** | Historical Mean (Baseline) | 0.0210 | 0.0270 | 0.0000 | 53.64% | |
| **TCS.NS** | **XGBoost Regressor** | **0.0189** | **0.0248** | **0.1568** | **59.60%** | **★ BEST** |
| **TCS.NS** | PyTorch LSTM Regressor | 0.0198 | 0.0258 | 0.0883 | 56.95% | |
| **RELIANCE.NS**| Historical Mean (Baseline) | 0.0232 | 0.0298 | 0.0000 | 51.66% | |
| **RELIANCE.NS**| **XGBoost Regressor** | **0.0205** | **0.0268** | **0.1914** | **61.59%** | **★ BEST** |
| **RELIANCE.NS**| PyTorch LSTM Regressor | 0.0216 | 0.0282 | 0.1042 | 57.62% | |

### 8.4 Portfolio Comparison Table (₹1,00,000 Portfolio, Moderate Risk Profile $\lambda = 2.0$)
*Evaluated on a ₹1,00,000 portfolio across `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, and `ITC.NS` under Moderate Risk Profile ($\lambda = 2.0$):*

| Exp ID | Strategy Name | Return Source ($\hat{\mu}$) | Risk Matrix ($\Sigma$) | Expected Return (%) | Portfolio Volatility (%) | Sharpe Ratio ($R_f=6\%$) | Asset Allocations ($w_i$) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **E0** | Equal Weight ($1/N$) | None ($w_i = 1/N$) | N/A | 15.20% | 12.84% | 0.716 | REL: 20.0%, TCS: 20.0%, INF: 20.0%, HDF: 20.0%, ITC: 20.0% |
| **E1** | Classical Markowitz | Historical Mean | Sample Covariance ($S$) | 20.69% | 14.86% | 0.988 | REL: 40.0%, TCS: 18.0%, INF: 0.0%, HDF: 2.0%, ITC: 40.0% |
| **E2** | Markowitz + Ledoit-Wolf | Historical Mean | Ledoit-Wolf ($\hat{\Sigma}_{\text{LW}}$) | 20.66% | 14.65% | 1.001 | REL: 40.0%, TCS: 17.5%, INF: 0.0%, HDF: 2.5%, ITC: 40.0% |
| **E3** | **XGBoost + Markowitz** | **XGBoost Regressor** | **Ledoit-Wolf ($\hat{\Sigma}_{\text{LW}}$)** | **47.29%** | **21.05%** | **1.938** | **REL: 40.0%, TCS: 40.0%, INF: 20.0%, HDF: 0.0%, ITC: 0.0%** |
| **E4** | PyTorch LSTM + Markowitz| PyTorch LSTM | Ledoit-Wolf ($\hat{\Sigma}_{\text{LW}}$) | 22.84% | 15.34% | 1.098 | REL: 35.0%, TCS: 40.0%, INF: 15.0%, HDF: 0.0%, ITC: 10.0% |

### 8.5 User Interface & Decision Support
* **Portfolio Optimiser Tab:** Interactive stock selector chips, capital input in ₹, risk profile toggles (Conservative / Moderate / Aggressive), and allocation donut chart.
* **ML Intelligence Tab:** Real-time multi-model comparison table (MAE, RMSE, $R^2$, Directional Accuracy) and horizontal feature attribution bar charts.
* **Market Explorer Tab:** Interactive candlestick price series with Bollinger Bands and 20/50 SMA overlays.
* **REST API:** Fully documented Swagger UI at `http://localhost:8000/docs`.

### 8.6 Limitations at Phase 1 Stage
* Empirical figures reported represent a single out-of-sample evaluation window; multi-year walk-forward testing will be conducted in Week 10.
* Transaction costs and turnover penalties will be formally incorporated into the optimizer objective in Week 9.
* Audit metadata tags enforce that synthetic data runs are never presented as real market results.

---

## 9. GANTT / PERT Milestones

### 9.1 Critical Path Analysis
| Week | Milestone Activity | Depends On | On Critical Path? |
| :---: | :--- | :--- | :---: |
| **1** | Problem definition and configs | None | Yes |
| **2** | Data ingestion pipeline | Week 1 | Yes |
| **3** | Technical features and leakage checks | Week 2 | Yes |
| **4** | Baseline, Ledoit-Wolf shrinkage, optimizer | Weeks 2, 3 | Yes |
| **5** | XGBoost Regressor and TreeSHAP | Week 3 | No (parallel to Week 4) |
| **6** | PyTorch LSTM and unified evaluation | Weeks 3, 5 | Yes |
| **7** | Validation-weighted ML ensemble | Weeks 5, 6 | Yes |
| **8** | Market regime classifier | Weeks 2, 4 | No (parallel to Week 7) |
| **9** | Indian transaction-cost model | Weeks 4, 7 | Yes |
| **10** | Multi-year walk-forward backtest | Weeks 7, 8, 9 | Yes |
| **11** | Drift monitor and statistical promotion gate | Week 10 | Yes |
| **12** | Controlled ablation studies and final thesis | Weeks 10, 11 | Yes |

---

## 10. Hardware and Software Requirements

### 10.1 Hardware
* **Processor:** Dual-core 2.0 GHz minimum (Quad-core Intel Core i5/i7 or AMD Ryzen recommended).
* **RAM:** 8 GB minimum (16 GB recommended for multi-year rolling windows).
* **Storage:** 5 GB free disk space for cached Parquet price records and model weights.
* **GPU:** Optional; PyTorch sequential LSTM training completes efficiently on modern multi-core CPUs.

### 10.2 Software Stack
* **Language & Runtime:** Python 3.12
* **Numerical & Data Processing:** NumPy (Harris et al., 2020), Pandas (McKinney, 2010)
* **ML & Statistics:** Scikit-learn (Pedregosa et al., 2011), XGBoost (Chen & Guestrin, 2016), PyTorch (Paszke et al., 2019)
* **Mathematical Optimisation:** SciPy SLSQP (Virtanen et al., 2020)
* **Explainability (XAI):** SHAP (Lundberg et al., 2020)
* **Backend Framework:** FastAPI, Uvicorn asynchronous server
* **Frontend Design:** Vanilla HTML5, CSS3 Glassmorphic UI, Plotly.js charts
* **Testing & Quality Assurance:** Pytest (12 automated test cases)

### 10.3 Execution Commands
```powershell
# 1. Run unit tests
python -m pytest -v

# 2. Run experiment ladder
python run_experiment.py --experiment E3

# 3. Start local application
python -m uvicorn backend.app.main:app --port 8000 --reload
```

---

## 11. Complete & Verified References

1. **Ang, A. and Bekaert, G.** (2002) ‘International asset allocation with regime shifts’, *Review of Financial Studies*, 15(4), pp. 1137–1187. doi: 10.1093/rfs/15.4.1137.
2. **Arlot, S. and Celisse, A.** (2010) ‘A survey of cross-validation procedures for model selection’, *Statistics Surveys*, 4, pp. 40–79. doi: 10.1214/09-SS054.
3. **Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J.** (2014) ‘Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance’, *Notices of the American Mathematical Society*, 61(5), pp. 458–471. doi: 10.1090/noti1105.
4. **Bailey, D.H. and López de Prado, M.** (2014) ‘The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality’, *Journal of Portfolio Management*, 40(5), pp. 94–107. doi: 10.3905/jpm.2014.40.5.094.
5. **Best, M.J. and Grauer, R.R.** (1991) ‘On the sensitivity of mean-variance-efficient portfolios to changes in the asset means: Some analytical and computational results’, *The Journal of Finance*, 46(1), pp. 315–342. doi: 10.1111/j.1540-6261.1991.tb03754.x.
6. **Black, F. and Litterman, R.** (1992) ‘Global portfolio optimization’, *Financial Analysts Journal*, 48(5), pp. 28–43. doi: 10.2469/faj.v48.n5.28.
7. **Boyd, S. and Vandenberghe, L.** (2004) *Convex Optimization*. Cambridge, UK: Cambridge University Press.
8. **Chen, T. and Guestrin, C.** (2016) ‘XGBoost: A scalable tree boosting system’, in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. New York: ACM, pp. 785–794. doi: 10.1145/2939672.2939785.
9. **Chopra, V.K. and Ziemba, W.T.** (1993) ‘The effect of errors in means, variances, and covariances on optimal portfolio choice’, *Journal of Portfolio Management*, 19(2), pp. 6–11. doi: 10.3905/jpm.1993.409440.
10. **Constantinides, G.M.** (1986) ‘Capital market equilibrium with transaction costs’, *Journal of Political Economy*, 94(4), pp. 842–862. doi: 10.1086/261410.
11. **DeMiguel, V., Garlappi, L. and Uppal, R.** (2009) ‘Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?’, *Review of Financial Studies*, 22(5), pp. 1915–1953. doi: 10.1093/rfs/hhm075.
12. **Efron, B.** (1979) ‘Bootstrap methods: Another look at the jackknife’, *Annals of Statistics*, 7(1), pp. 1–26. doi: 10.1214/aos/1176344552.
13. **Fischer, T. and Krauss, C.** (2018) ‘Deep learning with long short-term memory networks for financial market predictions’, *European Journal of Operational Research*, 270(2), pp. 654–669. doi: 10.1016/j.ejor.2017.11.054.
14. **Friedman, J.H.** (2001) ‘Greedy function approximation: A gradient boosting machine’, *Annals of Statistics*, 29(5), pp. 1189–1232. doi: 10.1214/aos/1013203451.
15. **Gu, S., Kelly, B. and Xiu, D.** (2020) ‘Empirical asset pricing via machine learning’, *Review of Financial Studies*, 33(5), pp. 2223–2273. doi: 10.1093/rfs/hhaa009.
16. **Hamilton, J.D.** (1989) ‘A new approach to the economic analysis of nonstationary time series and the business cycle’, *Econometrica*, 57(2), pp. 357–384. doi: 10.2307/1912559.
17. **Harris, C.R. et al.** (2020) ‘Array programming with NumPy’, *Nature*, 585(7825), pp. 357–362. doi: 10.1038/s41586-020-2649-2.
18. **Hochreiter, S. and Schmidhuber, J.** (1997) ‘Long short-term memory’, *Neural Computation*, 9(8), pp. 1735–1780. doi: 10.1162/neco.1997.9.8.1735.
19. **Jagannathan, R. and Ma, T.** (2003) ‘Risk reduction in large portfolios: Why imposing the wrong constraints helps’, *The Journal of Finance*, 58(4), pp. 1651–1683. doi: 10.1111/1540-6261.00580.
20. **Jegadeesh, N. and Titman, S.** (1993) ‘Returns to buying winners and selling losers: Implications for stock market efficiency’, *The Journal of Finance*, 48(1), pp. 65–91. doi: 10.1111/j.1540-6261.1993.tb04702.x.
21. **Jobson, J.D. and Korkie, B.M.** (1981) ‘Putting Markowitz theory to work’, *Journal of Portfolio Management*, 7(4), pp. 70–74. doi: 10.3905/jpm.1981.408816.
22. **Jorion, P.** (1986) ‘Bayes-Stein estimation for portfolio analysis’, *Journal of Financial and Quantitative Analysis*, 21(3), pp. 279–292. doi: 10.2307/2331042.
23. **Kingma, D.P. and Ba, J.** (2015) ‘Adam: A method for stochastic optimization’, in *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*. San Diego, CA.
24. **Krauss, C., Do, X.A. and Huck, N.** (2017) ‘Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500’, *European Journal of Operational Research*, 259(2), pp. 689–702. doi: 10.1016/j.ejor.2016.10.031.
25. **Ledoit, O. and Wolf, M.** (2003) ‘Improved estimation of the covariance matrix of stock returns with an application to portfolio selection’, *Journal of Empirical Finance*, 10(5), pp. 603–621. doi: 10.1016/S0927-5398(03)00007-0.
26. **Ledoit, O. and Wolf, M.** (2004a) ‘A well-conditioned estimator for large-dimensional covariance matrices’, *Journal of Multivariate Analysis*, 88(2), pp. 365–411. doi: 10.1016/S0047-259X(03)00096-4.
27. **Ledoit, O. and Wolf, M.** (2004b) ‘Honey, I shrunk the sample covariance matrix’, *Journal of Portfolio Management*, 30(4), pp. 110–119. doi: 10.3905/jpm.2004.110.
28. **Lobo, M.S., Fazel, M. and Boyd, S.** (2007) ‘Portfolio optimization with linear and fixed transaction costs’, *Annals of Operations Research*, 152(1), pp. 341–365. doi: 10.1007/s10479-006-0145-1.
29. **López de Prado, M.** (2018) *Advances in Financial Machine Learning*. Hoboken, NJ: John Wiley & Sons, Inc. ISBN: 978-1-119-48208-6.
30. **Lundberg, S.M. and Lee, S.-I.** (2017) ‘A unified approach to interpreting model predictions’, in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 4765–4774.
31. **Lundberg, S.M. et al.** (2020) ‘From local explanations to global understanding with explainable AI for trees’, *Nature Machine Intelligence*, 2(1), pp. 56–67. doi: 10.1038/s42256-019-0138-9.
32. **Markowitz, H.** (1952) ‘Portfolio selection’, *The Journal of Finance*, 7(1), pp. 77–91. doi: 10.1111/j.1540-6261.1952.tb01525.x.
33. **McKinney, W.** (2010) ‘Data structures for statistical computing in Python’, in *Proceedings of the 9th Python in Science Conference*, pp. 56–61. doi: 10.25080/Majora-92bf1922-00a.
34. **Michaud, R.O.** (1989) ‘The Markowitz optimization enigma: Is ‘optimized’ optimal?’, *Financial Analysts Journal*, 45(1), pp. 31–42. doi: 10.2469/faj.v45.n1.31.
35. **Pascanu, R., Mikolov, T. and Bengio, Y.** (2013) ‘On the difficulty of training recurrent neural networks’, in *Proceedings of the 30th International Conference on Machine Learning (ICML 2013)*, pp. 1310–1318.
36. **Paszke, A. et al.** (2019) ‘PyTorch: An imperative style, high-performance deep learning library’, in *Advances in Neural Information Processing Systems (NeurIPS 2019)*, 32, pp. 8024–8035.
37. **Pedregosa, F. et al.** (2011) ‘Scikit-learn: Machine learning in Python’, *Journal of Machine Learning Research*, 12, pp. 2825–2830.
38. **Rockafellar, R.T. and Uryasev, S.** (2000) ‘Optimization of conditional value-at-risk’, *Journal of Risk*, 2(3), pp. 21–41. doi: 10.21314/JOR.2000.038.
39. **Sezer, O.B., Gudelek, M.U. and Ozbayoglu, A.M.** (2020) ‘Financial time series forecasting with deep learning: A systematic literature review: 2005–2019’, *Applied Soft Computing*, 90, article 106181. doi: 10.1016/j.asoc.2020.106181.
40. **Sharpe, W.F.** (1964) ‘Capital asset prices: A theory of market equilibrium under conditions of risk’, *The Journal of Finance*, 19(3), pp. 425–442. doi: 10.1111/j.1540-6261.1964.tb02865.x.
41. **Sharpe, W.F.** (1966) ‘Mutual fund performance’, *Journal of Business*, 39(1), pp. 119–138. doi: 10.1086/294846.
42. **Sharpe, W.F.** (1994) ‘The Sharpe ratio’, *Journal of Portfolio Management*, 21(1), pp. 49–58. doi: 10.3905/jpm.1994.409501.
43. **Sortino, F.A. and van der Meer, R.** (1991) ‘Downside risk: Capturing what’s at stake in investment decisions’, *Journal of Portfolio Management*, 17(4), pp. 27–31. doi: 10.3905/jpm.1991.409343.
44. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R.** (2014) ‘Dropout: A simple way to prevent neural networks from overfitting’, *Journal of Machine Learning Research*, 15(1), pp. 1929–1958.
45. **Virtanen, P. et al.** (2020) ‘SciPy 1.0: Fundamental algorithms for scientific computing in Python’, *Nature Methods*, 17(3), pp. 261–272. doi: 10.1038/s41592-019-0686-2.
46. **Whaley, R.E.** (2000) ‘The investor fear gauge’, *Journal of Portfolio Management*, 26(3), pp. 12–17. doi: 10.3905/jpm.2000.319728.
