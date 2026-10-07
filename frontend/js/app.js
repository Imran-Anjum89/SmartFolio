/**
 * SmartFolio - Core Application Controller & Router
 */

const API_BASE = "/api";

const state = {
    universe: [],
    selectedSymbols: ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"],
    activeTab: "tab-optimizer",
    currentPortfolio: null,
    championVersion: "v1.0"
};

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadUniverse();
    initExplainerModal();
    initTickerRibbon();
});

function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));

            tab.classList.add("active");
            const targetTab = tab.getAttribute("data-tab");
            const contentEl = document.getElementById(targetTab);
            if (contentEl) {
                contentEl.classList.add("active");
                state.activeTab = targetTab;

                // Auto-load tab data if first time
                if (targetTab === "tab-ml") loadMLComparison();
                if (targetTab === "tab-explorer") loadExplorerData();
            }
        });
    });
}

async function loadUniverse() {
    try {
        const res = await fetch(`${API_BASE}/stocks/universe`);
        const data = await res.json();
        state.universe = data.assets || [];

        const chipsContainer = document.getElementById("universe-chips");
        if (!chipsContainer) return;
        chipsContainer.innerHTML = "";

        state.universe.forEach(asset => {
            const chip = document.createElement("div");
            const isSelected = state.selectedSymbols.includes(asset.symbol);
            const cleanSym = asset.symbol.replace(".NS", "");
            const monogram = cleanSym.slice(0, 2).toUpperCase();
            const sectorShort = asset.sector.split("/")[0].trim();

            chip.className = `chip sw-asset-card ${isSelected ? "selected" : ""}`;
            chip.setAttribute("data-symbol", asset.symbol);
            chip.setAttribute("tabindex", "0");
            chip.setAttribute("role", "checkbox");
            chip.setAttribute("aria-checked", isSelected ? "true" : "false");
            chip.innerHTML = `
                <div class="sw-asset-avatar" data-symbol="${cleanSym}">${monogram}</div>
                <div class="sw-asset-details">
                    <span class="sw-asset-symbol">${cleanSym}</span>
                    <span class="sw-asset-sector">${sectorShort}</span>
                </div>
                <div class="sw-asset-check" aria-hidden="true">✓</div>
            `;

            const toggleChip = () => {
                if (state.selectedSymbols.includes(asset.symbol)) {
                    if (state.selectedSymbols.length <= 2) {
                        alert("Please maintain at least 2 assets for portfolio optimization.");
                        return;
                    }
                    state.selectedSymbols = state.selectedSymbols.filter(s => s !== asset.symbol);
                    chip.classList.remove("selected");
                    chip.setAttribute("aria-checked", "false");
                } else {
                    state.selectedSymbols.push(asset.symbol);
                    chip.classList.add("selected");
                    chip.setAttribute("aria-checked", "true");
                }
            };

            chip.addEventListener("click", toggleChip);
            chip.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggleChip();
                }
            });

            chipsContainer.appendChild(chip);
        });
    } catch (err) {
        console.error("Failed to load stock universe:", err);
    }
}

function initTickerRibbon() {
    const tickerItems = document.querySelectorAll(".sw-ticker-item");
    tickerItems.forEach(item => {
        item.addEventListener("click", () => {
            const sym = item.getAttribute("data-symbol");
            if (!sym || sym === "NIFTY 50") return;

            const tabBtn = document.getElementById("btn-tab-explorer");
            if (tabBtn) tabBtn.click();

            const select = document.getElementById("explorer-symbol-select");
            if (select) {
                select.value = sym;
                const loadBtn = document.getElementById("btn-load-explorer");
                if (loadBtn) loadBtn.click();
            }
        });
    });
}

function initExplainerModal() {
    const modal = document.getElementById("modal-explainer");
    const closeBtn = document.getElementById("modal-close-btn");
    if (closeBtn && modal) {
        closeBtn.addEventListener("click", () => modal.classList.remove("open"));
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.classList.remove("open");
        });
    }
}

window.showExplainModal = function(symbol, explanationList) {
    const modal = document.getElementById("modal-explainer");
    const titleEl = document.getElementById("modal-title");
    const bodyEl = document.getElementById("modal-body-content");
    if (!modal || !bodyEl) return;

    titleEl.innerText = `Decision Rationale: ${symbol}`;
    let html = "<ul>";
    if (Array.isArray(explanationList) && explanationList.length > 0) {
        explanationList.forEach(item => {
            html += `<li>${item}</li>`;
        });
    } else {
        html += "<li>Optimal risk-adjusted balance within portfolio constraints.</li>";
    }
    html += "</ul>";
    bodyEl.innerHTML = html;
    modal.classList.add("open");
};
