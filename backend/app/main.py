"""
SmartFolio - Intelligent Investment Decision-Support System.
Main FastAPI application entrypoint.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.database.connection import init_db
from backend.app.api import stocks, predictions, portfolios

# Initialize Database tables
init_db()

app = FastAPI(
    title="SmartFolio API",
    description="Intelligent Investment Decision-Support System with ML Forecasting & Ledoit-Wolf Markowitz Optimization (Weeks 1-6)",
    version="1.0"
)

# Enable CORS for browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers (Weeks 1 to 6)
app.include_router(stocks.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(portfolios.router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "SmartFolio", "version": "1.0"}

# Serve Frontend static assets
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
