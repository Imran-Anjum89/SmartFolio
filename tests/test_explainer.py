"""
Unit Tests for Allocation Explainability Engine (Gap 7).
"""
import pytest
import numpy as np
from backend.app.services.explainer_service import AllocationExplainerService

def test_allocation_explainability():
    explainer = AllocationExplainerService()
    symbols = ["RELIANCE.NS", "TCS.NS"]
    allocations = [
        {"symbol": "RELIANCE.NS", "weight": 0.40, "investment_amount": 40000.0, "expected_return": 0.25},
        {"symbol": "TCS.NS", "weight": 0.20, "investment_amount": 20000.0, "expected_return": 0.12}
    ]
    expected_returns = {"RELIANCE.NS": 0.25, "TCS.NS": 0.12}
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.03]])

    results = explainer.explain_allocations(allocations, expected_returns, cov_matrix, symbols, w_max=0.40)
    assert len(results) == 2
    for item in results:
        assert "explanation" in item
        assert len(item["explanation"]) > 0

    # Max weight constraint check for RELIANCE (weight=0.40)
    reliance_expl = results[0]["explanation"]
    assert any("ceiling" in exp or "Maximum" in exp for exp in reliance_expl)
