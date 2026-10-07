"""
Database Connection and Session Management for SmartFolio.
Defaults to SQLite for zero-setup local development, with full PostgreSQL compatibility.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import DATABASE_URL

# SQLite connect_args require check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency generator providing a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all database tables."""
    from backend.app.models.db_models import PortfolioRecord, PortfolioAssetRecord, StockPredictionRecord
    Base.metadata.create_all(bind=engine)
