"""
SQLAlchemy ORM Database Models for SmartFolio.
"""
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.database.connection import Base

class PortfolioRecord(Base):
    __tablename__ = "portfolios"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), default="My Portfolio")
    investment_amount = Column(Float, nullable=False)
    risk_profile = Column(String(32), nullable=False)
    strategy = Column(String(32), nullable=False)
    expected_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    assets = relationship("PortfolioAssetRecord", back_populates="portfolio", cascade="all, delete-orphan")

class PortfolioAssetRecord(Base):
    __tablename__ = "portfolio_assets"

    id = Column(String(64), primary_key=True, index=True)
    portfolio_id = Column(String(64), ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String(32), nullable=False)
    weight = Column(Float, nullable=False)
    investment_amount = Column(Float, nullable=False)
    predicted_return = Column(Float)
    explanation = Column(Text)

    portfolio = relationship("PortfolioRecord", back_populates="assets")

class StockPredictionRecord(Base):
    __tablename__ = "stock_predictions"

    id = Column(String(64), primary_key=True, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    prediction_date = Column(String(32), nullable=False)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)
    predicted_return = Column(Float, nullable=False)
    actual_return = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)
    squared_error = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
