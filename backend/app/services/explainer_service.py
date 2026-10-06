"""
Allocation Explainability Engine for SmartFolio.
Generates transparent, rule-based reasoning for every asset weight recommendation (Gap 7).
"""
from typing import Dict, List, Any
import numpy as np

class AllocationExplainerService:
    def explain_allocations(
        self,
        allocations: List[Dict[str, Any]],
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
        symbols: List[str],
        w_max: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        Generate rule-based explanation bullet points for each asset allocation.
        """
        mean_exp_ret = float(np.mean(list(expected_returns.values())))
        diag_vols = np.sqrt(np.diag(cov_matrix))
        mean_vol = float(np.mean(diag_vols))

        explained_allocations = []
        for item in allocations:
            sym = item["symbol"]
            w = item["weight"]
            idx = symbols.index(sym) if sym in symbols else 0
            asset_ret = expected_returns.get(sym, 0.0)
            asset_vol = float(diag_vols[idx]) if idx < len(diag_vols) else mean_vol

            reasons = []

            # 1. Expected Return Driver
            if asset_ret > mean_exp_ret * 1.15:
                reasons.append(f"Strong predicted annual return (+{asset_ret * 100:.1f}%), exceeding universe average.")
            elif asset_ret < mean_exp_ret * 0.85:
                reasons.append(f"Subdued forecasted return (+{asset_ret * 100:.1f}%), limiting allocation weight.")
            else:
                reasons.append(f"Consistent return profile (+{asset_ret * 100:.1f}%) in line with market expectation.")

            # 2. Risk & Volatility Factor
            if asset_vol < mean_vol * 0.90:
                reasons.append(f"Low individual volatility ({asset_vol * 100:.1f}%), serving as a portfolio stabilizer.")
            elif asset_vol > mean_vol * 1.20:
                reasons.append(f"Elevated individual volatility ({asset_vol * 100:.1f}%), which optimizer throttled to control portfolio risk.")
            else:
                reasons.append(f"Balanced risk contribution ({asset_vol * 100:.1f}% annualized volatility).")

            # 3. Constraint and Allocation Behavior
            if np.isclose(w, w_max, atol=0.01):
                reasons.append(f"Maximum allocation safety ceiling ({w_max * 100:.0f}%) reached.")
            elif w > 0.20:
                reasons.append("High capital allocation awarded due to favorable risk-adjusted Sharpe contribution.")
            elif w < 0.05:
                reasons.append("Marginal allocation due to better risk-adjusted alternatives in other sectors.")
            else:
                reasons.append("Healthy diversification weight within prescribed bounds.")

            item_copy = dict(item)
            item_copy["explanation"] = reasons
            explained_allocations.append(item_copy)

        return explained_allocations
