/**
 * SmartFolio - Machine Learning Model Comparison Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnEval = document.getElementById("btn-evaluate-ml");
    if (btnEval) {
        btnEval.addEventListener("click", loadMLComparison);
    }
});

async function loadMLComparison() {
    const select = document.getElementById("ml-symbol-select");
    const symbol = select ? select.value : "TCS.NS";
    const btn = document.getElementById("btn-evaluate-ml");
    if (btn) {
        btn.disabled = true;
        btn.innerText = "Training & Evaluating...";
    }

    try {
        const res = await fetch(`${API_BASE}/predictions/compare/${symbol}`);
        if (!res.ok) throw new Error("Failed to evaluate models for " + symbol);
        const data = await res.json();

        // 1. Render Metrics Table
        const tbody = document.querySelector("#table-model-eval tbody");
        tbody.innerHTML = "";

        const modelDisplayNames = {
            "baseline": "Historical Mean (Baseline)",
            "xgboost": "XGBoost Regressor",
            "lstm": "PyTorch LSTM Regressor"
        };

        for (const [mKey, metrics] of Object.entries(data.metrics)) {
            const isBest = (mKey === data.best_model);
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <strong>${modelDisplayNames[mKey] || mKey}</strong>
                    ${isBest ? '<span class="badge-tag" style="margin-left:8px;background:rgba(16,185,129,0.2);color:#34d399;border-color:rgba(16,185,129,0.4)">BEST</span>' : ''}
                </td>
                <td style="font-family:var(--font-mono)">${metrics.mae.toFixed(4)}</td>
                <td style="font-family:var(--font-mono)"><strong>${metrics.rmse.toFixed(4)}</strong></td>
                <td style="font-family:var(--font-mono)">${metrics.r2.toFixed(3)}</td>
                <td style="font-family:var(--font-mono);color:#38bdf8">${(metrics.directional_accuracy * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        }

        // 2. Render Feature Importance Horizontal Bar Chart
        renderFeatureImportanceChart(data.feature_importance);

    } catch (err) {
        console.error("ML evaluation failed:", err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerText = "Evaluate Models";
        }
    }
}

function renderFeatureImportanceChart(importances) {
    const sortedEntries = Object.entries(importances).sort((a, b) => a[1] - b[1]);
    const features = sortedEntries.map(e => e[0].toUpperCase());
    const values = sortedEntries.map(e => e[1]);

    const trace = {
        type: "bar",
        x: values,
        y: features,
        orientation: "h",
        marker: {
            color: values,
            colorscale: "Tealgrn"
        }
    };

    const layout = {
        margin: { l: 120, r: 20, t: 20, b: 30 },
        height: 280,
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        xaxis: {
            title: "Importance Score",
            color: "#94a3b8",
            gridcolor: "rgba(255, 255, 255, 0.05)"
        },
        yaxis: {
            color: "#94a3b8",
            automargin: true
        }
    };

    Plotly.newPlot("plot-feature-importance", [trace], layout, { responsive: true, displayModeBar: false });
}
