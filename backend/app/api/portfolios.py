"""
Portfolio Optimization and Allocation Endpoints for SmartFolio.
"""
import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from backend.app.schemas.schemas import PortfolioOptimizeRequest, PortfolioOptimizeResponse
from backend.app.services.data_service import MarketDataService
from backend.app.services.ml_service import MLService
from backend.app.services.risk_service import RiskService
from backend.app.services.optimizer_service import PortfolioOptimizerService
from backend.app.services.explainer_service import AllocationExplainerService
from backend.app.database.connection import get_db
from backend.app.models.db_models import PortfolioRecord, PortfolioAssetRecord

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])

data_service = MarketDataService()
ml_service = MLService()
risk_service = RiskService()
optimizer_service = PortfolioOptimizerService()
explainer_service = AllocationExplainerService()

@router.post("/optimize", response_model=PortfolioOptimizeResponse)
def optimize_portfolio_endpoint(
    req: PortfolioOptimizeRequest,
    db: Session = Depends(get_db)
):
    """
    Run Markowitz Portfolio Optimization using ML return forecasts and Ledoit-Wolf covariance shrinkage.
    Generates rule-based allocation explainability and stores portfolio in database.
    """
    symbols = req.symbols
    if len(symbols) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 assets for portfolio optimization.")

    try:
        # 1. Ingest stock price data
        price_series_dict = {}
        symbols_raw_data = {}
        for s in symbols:
            df, _ = data_service.fetch_stock_data(s)
            symbols_raw_data[s] = df
            if "date" in df.columns and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                price_series_dict[s] = df.set_index("date")["close"]

        # 2. Risk & Covariance estimation (Ledoit-Wolf)
        returns_df = risk_service.compute_asset_returns_matrix(price_series_dict)
        cov_lw, _ = risk_service.compute_ledoit_wolf_covariance(returns_df)

        # 3. Forecast expected returns mu_hat via selected model
        exp_returns = ml_service.predict_expected_returns(symbols_raw_data, model_name=req.model_name)

        # 4. Solve constrained Markowitz optimization
        opt_result = optimizer_service.optimize_portfolio(
            expected_returns=exp_returns,
            cov_matrix=cov_lw,
            strategy=req.strategy,
            risk_profile=req.risk_profile,
            total_capital=req.total_capital
        )

        # 5. Generate rule-based allocation explainability
        explained_allocations = explainer_service.explain_allocations(
            allocations=opt_result["allocations"],
            expected_returns=exp_returns,
            cov_matrix=cov_lw,
            symbols=symbols
        )

        portfolio_id = str(uuid.uuid4())[:8]

        # 6. Save to Database
        try:
            port_record = PortfolioRecord(
                id=portfolio_id,
                name=f"{req.risk_profile.capitalize()} Portfolio",
                investment_amount=req.total_capital,
                risk_profile=req.risk_profile,
                strategy=req.strategy,
                expected_return=opt_result["expected_portfolio_return"],
                volatility=opt_result["portfolio_volatility"],
                sharpe_ratio=opt_result["sharpe_ratio"]
            )
            db.add(port_record)

            for item in explained_allocations:
                asset_rec = PortfolioAssetRecord(
                    id=str(uuid.uuid4())[:8],
                    portfolio_id=portfolio_id,
                    symbol=item["symbol"],
                    weight=item["weight"],
                    investment_amount=item["investment_amount"],
                    predicted_return=item["expected_return"],
                    explanation="; ".join(item["explanation"]) if item.get("explanation") else ""
                )
                db.add(asset_rec)
            db.commit()
        except Exception as db_err:
            db.rollback()

        return {
            "portfolio_id": portfolio_id,
            "strategy": opt_result["strategy"],
            "risk_profile": opt_result["risk_profile"],
            "risk_aversion": opt_result["risk_aversion"],
            "total_capital": opt_result["total_capital"],
            "expected_portfolio_return": opt_result["expected_portfolio_return"],
            "portfolio_volatility": opt_result["portfolio_volatility"],
            "sharpe_ratio": opt_result["sharpe_ratio"],
            "model_used": req.model_name,
            "model_version": ml_service.current_model_version,
            "allocations": explained_allocations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

@router.get("/saved")
def get_saved_portfolios(db: Session = Depends(get_db)):
    """Retrieve list of saved portfolios from database."""
    portfolios = db.query(PortfolioRecord).order_by(PortfolioRecord.created_at.desc()).limit(10).all()
    results = []
    for p in portfolios:
        results.append({
            "id": p.id,
            "name": p.name,
            "investment_amount": p.investment_amount,
            "risk_profile": p.risk_profile,
            "strategy": p.strategy,
            "expected_return": p.expected_return,
            "volatility": p.volatility,
            "sharpe_ratio": p.sharpe_ratio,
            "created_at": p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else ""
        })
    return results
