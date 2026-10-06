/**
 * SmartFolio - Portfolio Optimizer Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnOptimize = document.getElementById("btn-optimize");
    if (btnOptimize) {
        btnOptimize.addEventListener("click", runPortfolioOptimization);
    }
});

async function runPortfolioOptimization() {
    const consentCheckbox = document.getElementById("input-consent-optimizer");
    const consentWarning = document.getElementById("consent-warning-msg");
    if (consentCheckbox && !consentCheckbox.checked) {
        if (consentWarning) {
            consentWarning.style.display = "block";
            consentWarning.classList.add("shake-animation");
            setTimeout(() => consentWarning.classList.remove("shake-animation"), 600);
        }
        consentCheckbox.focus();
        return;
    }

    const btn = document.getElementById("btn-optimize");
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    btn.innerText = "⏳ Optimizing Portfolio...";

    try {
        const capital = parseFloat(document.getElementById("input-capital").value) || 100000;
        const strategy = document.getElementById("select-strategy").value;
        const modelName = document.getElementById("select-ml-model").value;
        const riskProfile = document.querySelector("input[name='risk-profile']:checked")?.value || "moderate";

        const payload = {
            symbols: state.selectedSymbols,
            total_capital: capital,
            strategy: strategy,
            risk_profile: riskProfile,
            model_name: modelName
        };

        const res = await fetch(`${API_BASE}/portfolios/optimize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Optimization failed");
        }

        const data = await res.json();
        state.currentPortfolio = data;
        renderOptimizationResults(data);

    } catch (err) {
        alert("Error generating portfolio: " + err.message);
    } finally {
        btn.disabled = false;
        btn.removeAttribute("aria-busy");
        btn.innerHTML = "<span>⚡ Generate Optimized Allocation</span>";
    }
}

function renderOptimizationResults(data) {
    // 1. Update KPI Cards & StockWise Hero Card
    const expReturnStr = `+${(data.expected_portfolio_return * 100).toFixed(2)}%`;
    const volStr = `${(data.portfolio_volatility * 100).toFixed(2)}%`;
    const sharpeStr = data.sharpe_ratio.toFixed(2);
    const capitalStr = `₹${data.total_capital.toLocaleString('en-IN')}`;

    document.getElementById("res-expected-return").innerText = expReturnStr;
    document.getElementById("res-volatility").innerText = volStr;
    document.getElementById("res-sharpe").innerText = sharpeStr;
    document.getElementById("res-capital").innerText = capitalStr;
    document.getElementById("res-model-badge").innerText = `Model: ${data.model_used.toUpperCase()} (${data.model_version})`;

    // Update StockWise Hero Portfolio Card
    const heroCap = document.getElementById("hero-total-capital");
    if (heroCap) heroCap.innerText = capitalStr;
    const heroRet = document.getElementById("hero-projected-return");
    if (heroRet) heroRet.innerText = `▲ ${expReturnStr} Exp. Return`;
    const heroSharpe = document.getElementById("hero-sharpe-score");
    if (heroSharpe) heroSharpe.innerText = `⚡ Sharpe ${sharpeStr}`;
    const heroVol = document.getElementById("hero-volatility-val");
    if (heroVol) heroVol.innerText = volStr;
    const heroModel = document.getElementById("hero-model-tag");
    if (heroModel) heroModel.innerText = `${data.model_used.toUpperCase()} ${data.model_version}`;

    // 2. Render Plotly Donut Chart with StockWise Palette
    const labels = data.allocations.map(a => a.symbol.replace(".NS", ""));
    const values = data.allocations.map(a => a.weight * 100);

    const donutData = [{
        values: values,
        labels: labels,
        type: "pie",
        hole: 0.64,
        textinfo: "label+percent",
        textposition: "inside",
        hoverinfo: "label+percent+value",
        marker: {
            colors: ["#00f090", "#00e5ff", "#8b5cf6", "#38bdf8", "#ffb800", "#ff4d6d", "#a3e635", "#2dd4bf"]
        }
    }];

    const donutLayout = {
        showlegend: true,
        legend: { orientation: "h", x: 0.1, y: -0.15, font: { color: "#94a3b8" } },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        margin: { t: 20, b: 20, l: 20, r: 20 },
        height: 290,
        annotations: [{
            font: { size: 14, color: "#ffffff", weight: "bold", family: "Plus Jakarta Sans" },
            showarrow: false,
            text: `Sharpe<br><b style="color:#00f090; font-size:18px">${data.sharpe_ratio.toFixed(2)}</b>`,
            x: 0.5,
            y: 0.5
        }]
    };

    Plotly.newPlot("plot-portfolio-donut", donutData, donutLayout, { responsive: true, displayModeBar: false });

    // 3. Render Asset Breakdown Table
    const tbody = document.querySelector("#table-allocations tbody");
    tbody.innerHTML = "";

    data.allocations.forEach(item => {
        const tr = document.createElement("tr");
        const expJson = JSON.stringify(item.explanation || []).replace(/"/g, '&quot;');

        tr.innerHTML = `
            <td><strong>${item.symbol.replace(".NS", "")}</strong></td>
            <td>${item.weight_percentage.toFixed(2)}%</td>
            <td style="font-family: var(--font-mono)">₹${item.investment_amount.toLocaleString('en-IN')}</td>
            <td class="${item.expected_return >= 0 ? 'text-green' : 'text-amber'}">+${(item.expected_return * 100).toFixed(2)}%</td>
            <td>
                <span class="explain-badge" onclick='showExplainModal("${item.symbol}", ${expJson})'>
                    🔍 View Rationale
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}
