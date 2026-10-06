"""
Machine Learning Model Prediction & Comparison Endpoints for SmartFolio.
"""
from fastapi import APIRouter, HTTPException
from backend.app.services.data_service import MarketDataService
from backend.app.services.ml_service import MLService
from backend.app.schemas.schemas import ModelComparisonResponse

router = APIRouter(prefix="/predictions", tags=["Predictions"])

data_service = MarketDataService()
ml_service = MLService()

@router.get("/compare/{symbol}", response_model=ModelComparisonResponse)
def compare_models_for_stock(symbol: str):
    """
    Train and evaluate Baseline, XGBoost, PyTorch LSTM, and Ensemble on the selected stock.
    Returns out-of-sample metrics (MAE, RMSE, R2, Directional Accuracy) and feature importances.
    """
    try:
        df_raw, _ = data_service.fetch_stock_data(symbol)
        bundle = ml_service.train_models_for_symbol(symbol, df_raw)
        return {
            "symbol": symbol,
            "version": bundle["version"],
            "best_model": bundle["best_model"],
            "metrics": bundle["metrics"],
            "feature_importance": bundle["feature_importance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed for {symbol}: {str(e)}")

@router.get("/expected-returns")
def get_expected_returns(model_name: str = "ensemble"):
    """
    Get 5-day expected return predictions (annualized) across the active universe.
    """
    try:
        symbols_data = {}
        for sym in data_service.symbols[:8]:
            df, _ = data_service.fetch_stock_data(sym)
            symbols_data[sym] = df

        exp_returns = ml_service.predict_expected_returns(symbols_data, model_name=model_name)
        return {
            "model_used": model_name,
            "model_version": ml_service.current_model_version,
            "expected_returns": exp_returns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
