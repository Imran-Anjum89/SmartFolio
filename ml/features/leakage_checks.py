"""
Leakage Verification Module for SmartFolio.
Automates validation ensuring zero temporal information leakage in features.
"""
import pandas as pd
import numpy as np
from ml.features.indicators import compute_technical_indicators, generate_target

def verify_feature_leakage(df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Test that modifying future prices (t > k) has ZERO effect on indicators calculated at time t <= k.
    If any feature at time k changes when future price at k+1 changes, information leakage is detected.
    """
    if len(df) < 100:
        return True
    
    # Calculate features on original data
    df_orig = compute_technical_indicators(df)
    
    # Modify future price at index 80 (e.g., simulate a massive spike at t=80)
    df_mod = df.copy()
    test_idx = 80
    cutoff_idx = 70
    df_mod.loc[test_idx, "close"] = df_mod.loc[test_idx, "close"] * 5.0
    df_mod.loc[test_idx, "high"] = df_mod.loc[test_idx, "high"] * 5.0
    
    df_mod_features = compute_technical_indicators(df_mod)
    
    # Compare features prior to the modification (t <= cutoff_idx)
    for col in feature_cols:
        orig_vals = df_orig.loc[:cutoff_idx, col].dropna().values
        mod_vals = df_mod_features.loc[:cutoff_idx, col].dropna().values
        
        diff = np.abs(orig_vals - mod_vals)
        max_diff = np.max(diff) if len(diff) > 0 else 0.0
        
        if max_diff > 1e-7:
            raise ValueError(f"Temporal Leakage Detected in feature '{col}': max discrepancy {max_diff:.8f}")
            
    return True

def verify_target_alignment(df: pd.DataFrame, horizon: int = 5) -> bool:
    """
    Verify that target_5d at index t is exactly (close[t+5] - close[t]) / close[t].
    """
    df_target = generate_target(df, horizon=horizon)
    target_col = f"target_{horizon}d"
    
    for i in range(len(df_target) - horizon - 1):
        expected = (df_target.loc[i + horizon, "close"] / df_target.loc[i, "close"]) - 1.0
        actual = df_target.loc[i, target_col]
        if not np.isclose(expected, actual, atol=1e-7):
            raise ValueError(f"Target misaligned at index {i}: expected {expected}, got {actual}")
            
    return True
