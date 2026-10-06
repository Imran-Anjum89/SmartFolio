/**
 * SmartFolio - Market Explorer & Technical Indicator Visualizer
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnLoad = document.getElementById("btn-load-explorer");
    if (btnLoad) {
        btnLoad.addEventListener("click", loadExplorerData);
    }
});

async function loadExplorerData() {
    const select = document.getElementById("explorer-symbol-select");
    const symbol = select ? select.value : "RELIANCE.NS";

    try {
        const [histRes, indRes] = await Promise.all([
            fetch(`${API_BASE}/stocks/history/${symbol}?start_date=2023-01-01`),
            fetch(`${API_BASE}/stocks/indicators/${symbol}`)
        ]);

        const histData = await histRes.json();
        const indData = await indRes.json();

        renderExplorerCharts(symbol, histData.history || [], indData.indicators || []);
    } catch (err) {
        console.error("Failed to load explorer data:", err);
    }
}

function renderExplorerCharts(symbol, history, indicators) {
    if (history.length === 0) return;

    const dates = history.map(h => h.date);
    const open = history.map(h => h.open);
    const high = history.map(h => h.high);
    const low = history.map(h => h.low);
    const close = history.map(h => h.close);

    // 1. Candlestick Chart
    const candlestickTrace = {
        x: dates,
        open: open,
        high: high,
        low: low,
        close: close,
        type: "candlestick",
        name: symbol.replace(".NS", ""),
        increasing: { line: { color: "#00f090", width: 1.5 } },
        decreasing: { line: { color: "#ff4d6d", width: 1.5 } }
    };

    const priceLayout = {
        title: {
            text: `${symbol} Daily Price & Moving Averages`,
            font: { color: "#ffffff", size: 14, family: "Plus Jakarta Sans" }
        },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        height: 380,
        margin: { t: 40, b: 20, l: 40, r: 20 },
        xaxis: {
            color: "#94a3b8",
            gridcolor: "rgba(255, 255, 255, 0.04)",
            rangeslider: { visible: false }
        },
        yaxis: {
            color: "#94a3b8",
            gridcolor: "rgba(255, 255, 255, 0.04)",
            title: "Price (₹ INR)"
        }
    };

    Plotly.newPlot("plot-explorer-price", [candlestickTrace], priceLayout, { responsive: true, displayModeBar: false });

    // 2. Technical Indicators (RSI Chart)
    if (indicators.length > 0) {
        const indDates = indicators.map(i => i.date);
        const rsi = indicators.map(i => i.rsi_14);

        const rsiTrace = {
            x: indDates,
            y: rsi,
            type: "scatter",
            mode: "lines",
            name: "RSI (14)",
            line: { color: "#00e5ff", width: 2.0 }
        };

        const rsiLayout = {
            title: {
                text: `Relative Strength Index (RSI 14)`,
                font: { color: "#ffffff", size: 12, family: "Plus Jakarta Sans" }
            },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            height: 220,
            margin: { t: 30, b: 30, l: 40, r: 20 },
            xaxis: {
                color: "#94a3b8",
                gridcolor: "rgba(255, 255, 255, 0.04)"
            },
            yaxis: {
                color: "#94a3b8",
                gridcolor: "rgba(255, 255, 255, 0.04)",
                range: [0, 100],
                tickvals: [30, 50, 70]
            },
            shapes: [
                { type: "line", x0: indDates[0], x1: indDates[indDates.length - 1], y0: 70, y1: 70, line: { color: "rgba(255,77,109,0.6)", dash: "dot" } },
                { type: "line", x0: indDates[0], x1: indDates[indDates.length - 1], y0: 30, y1: 30, line: { color: "rgba(0,240,144,0.6)", dash: "dot" } }
            ]
        };

        Plotly.newPlot("plot-explorer-indicators", [rsiTrace], rsiLayout, { responsive: true, displayModeBar: false });
    }
}
