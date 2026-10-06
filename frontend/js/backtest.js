/**
 * SmartFolio - Walk-Forward Backtesting Frontend Logic (E0 - E6)
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnBacktest = document.getElementById("btn-run-backtest");
    if (btnBacktest) {
        btnBacktest.addEventListener("click", runWalkForwardBacktest);
    }
});

async function runWalkForwardBacktest() {
    const btn = document.getElementById("btn-run-backtest");
    btn.disabled = true;
    btn.innerText = "⏳ Running Walk-Forward Simulation...";

    try {
        const payload = {
            symbols: state.selectedSymbols,
            start_date: "2021-01-01",
            rebalance_days: 21,
            lookback_window_days: 252
        };

        const res = await fetch(`${API_BASE}/backtesting/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Backtest failed");
        }

        const data = await res.json();
        renderBacktestResults(data);

    } catch (err) {
        alert("Error running backtest: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>▶ Run Walk-Forward Simulation</span>";
    }
}

function renderBacktestResults(data) {
    const dates = data.dates;
    const wealthData = data.cumulative_wealth;
    const metrics = data.metrics_summary;

    // 1. Render Multi-Line Plotly Chart with StockWise Palette
    const colorMap = {
        "E0_EqualWeight": "#64748b",
        "E1_TradMarkowitz": "#ffb800",
        "E2_Markowitz_LW": "#00e5ff",
        "E3_XGBoost": "#38bdf8",
        "E4_LSTM": "#d946ef",
        "E5_Ensemble": "#8b5cf6",
        "E6_SmartFolioAdaptive": "#00f090"
    };

    const labelMap = {
        "E0_EqualWeight": "E0: Equal Weight (1/N)",
        "E1_TradMarkowitz": "E1: Traditional Markowitz",
        "E2_Markowitz_LW": "E2: Markowitz + Ledoit-Wolf",
        "E3_XGBoost": "E3: XGBoost + Markowitz",
        "E4_LSTM": "E4: PyTorch LSTM + Markowitz",
        "E5_Ensemble": "E5: XGB-LSTM Ensemble",
        "E6_SmartFolioAdaptive": "E6: SmartFolio Adaptive (Proposed)"
    };

    const traces = [];
    for (const strat of data.strategies) {
        const isAdaptive = (strat === "E6_SmartFolioAdaptive");
        traces.push({
            x: dates,
            y: wealthData[strat],
            name: labelMap[strat] || strat,
            type: "scatter",
            mode: "lines",
            line: {
                color: colorMap[strat] || "#cbd5e1",
                width: isAdaptive ? 3.5 : 1.75,
                dash: strat === "E0_EqualWeight" ? "dot" : "solid"
            }
        });
    }

    const layout = {
        title: {
            text: "Cumulative Wealth Growth (Starting Base = 1.00)",
            font: { color: "#f8fafc", size: 14 }
        },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        height: 440,
        margin: { t: 40, b: 40, l: 40, r: 20 },
        legend: {
            orientation: "h",
            x: 0.05,
            y: -0.2,
            font: { color: "#94a3b8", size: 11 }
        },
        xaxis: {
            color: "#94a3b8",
            gridcolor: "rgba(255, 255, 255, 0.05)"
        },
        yaxis: {
            color: "#94a3b8",
            gridcolor: "rgba(255, 255, 255, 0.05)",
            title: "Portfolio Wealth Multiplier"
        }
    };

    Plotly.newPlot("plot-backtest-wealth", traces, layout, { responsive: true, displayModeBar: false });

    // 2. Render Statistical Bootstrap Significance Banner
    const stat = data.statistical_validation;
    if (stat) {
        const box = document.getElementById("bootstrap-box");
        box.style.display = "flex";
        document.getElementById("bb-delta-val").innerText = `${stat.delta_sharpe >= 0 ? '+' : ''}${stat.delta_sharpe.toFixed(2)}`;
        document.getElementById("bb-ci-val").innerText = `[${stat.confidence_interval_95[0].toFixed(2)}, ${stat.confidence_interval_95[1].toFixed(2)}]`;
        document.getElementById("bb-sig-val").innerText = stat.statistically_significant ? "Statistically Significant (p < 0.05)" : "Inconclusive Support";
        document.getElementById("bb-sig-val").className = stat.statistically_significant ? "val text-green" : "val text-amber";
    }

    // 3. Render Summary Metrics Table
    const tbody = document.querySelector("#table-backtest-summary tbody");
    tbody.innerHTML = "";

    data.strategies.forEach(strat => {
        const m = metrics[strat];
        const isAdaptive = (strat === "E6_SmartFolioAdaptive");
        const tr = document.createElement("tr");
        if (isAdaptive) tr.style.backgroundColor = "rgba(16, 185, 129, 0.08)";

        tr.innerHTML = `
            <td>
                <strong>${labelMap[strat] || strat}</strong>
                ${isAdaptive ? '<span class="badge-tag" style="margin-left:8px;background:rgba(16,185,129,0.2);color:#34d399;border-color:rgba(16,185,129,0.4)">PROPOSED</span>' : ''}
            </td>
            <td style="font-family:var(--font-mono);color:${m.cumulative_return >= 0 ? '#34d399' : '#f87171'}">+${(m.cumulative_return * 100).toFixed(1)}%</td>
            <td style="font-family:var(--font-mono)">+${(m.annualized_return * 100).toFixed(1)}%</td>
            <td style="font-family:var(--font-mono)">${(m.annualized_volatility * 100).toFixed(1)}%</td>
            <td style="font-family:var(--font-mono);font-weight:700" class="text-purple">${m.sharpe_ratio.toFixed(2)}</td>
            <td style="font-family:var(--font-mono)">${m.sortino_ratio.toFixed(2)}</td>
            <td style="font-family:var(--font-mono);color:#f87171">${(m.max_drawdown * 100).toFixed(1)}%</td>
            <td style="font-family:var(--font-mono)">${(m.average_turnover * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });
}
