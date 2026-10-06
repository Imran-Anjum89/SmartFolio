/**
 * SmartFolio - Dual Feedback Engine & Adaptive Retraining Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnRetrain = document.getElementById("btn-trigger-retrain");
    if (btnRetrain) {
        btnRetrain.addEventListener("click", triggerAdaptiveRetraining);
    }
});

async function loadFeedbackSummary() {
    try {
        const res = await fetch(`${API_BASE}/feedback/summary`);
        if (!res.ok) throw new Error("Failed to load feedback summary");
        const data = await res.json();

        document.getElementById("fb-champion-ver").innerText = data.current_champion_version;
        document.getElementById("fb-total-count").innerText = data.total_prediction_feedback_count;
        document.getElementById("fb-mae").innerText = `${(data.mae * 100).toFixed(2)}%`;
        document.getElementById("fb-promotions").innerText = data.promotions_count;

        // Update Header Badge
        const champBadge = document.getElementById("champion-badge");
        if (champBadge) champBadge.innerText = `Active Champion: ${data.current_champion_version} (Ensemble)`;

        // Render Recent Prediction Logs
        const tbody = document.querySelector("#table-feedback-logs tbody");
        if (data.recent_predictions && data.recent_predictions.length > 0) {
            tbody.innerHTML = "";
            data.recent_predictions.slice().reverse().forEach(log => {
                const tr = document.createElement("tr");
                const errColor = log.error >= 0 ? "text-green" : "text-amber";
                tr.innerHTML = `
                    <td style="font-family:var(--font-mono);font-size:0.75rem">${log.timestamp.slice(0, 19).replace("T", " ")}</td>
                    <td><strong>${log.symbol.replace(".NS", "")}</strong></td>
                    <td><span class="badge-tag">${log.model_version}</span></td>
                    <td style="font-family:var(--font-mono)">+${(log.predicted_return * 100).toFixed(2)}%</td>
                    <td style="font-family:var(--font-mono)">+${(log.actual_return * 100).toFixed(2)}%</td>
                    <td style="font-family:var(--font-mono)" class="${errColor}">${(log.error * 100).toFixed(2)}%</td>
                    <td style="font-family:var(--font-mono)">${(log.absolute_error * 100).toFixed(2)}%</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Error loading feedback summary:", err);
    }
}

async function triggerAdaptiveRetraining() {
    const btn = document.getElementById("btn-trigger-retrain");
    btn.disabled = true;
    btn.innerText = "⏳ Retraining Challenger...";

    try {
        const res = await fetch(`${API_BASE}/feedback/retrain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: "TCS.NS" })
        });

        if (!res.ok) throw new Error("Retraining failed");
        const report = await res.json();

        // Display Champion-Challenger Verdict Box
        const statusBox = document.getElementById("challenger-status-box");
        const badge = document.getElementById("verdict-badge");
        const title = document.getElementById("verdict-title");
        const reason = document.getElementById("verdict-reason");

        statusBox.style.display = "block";
        if (report.should_promote) {
            badge.className = "badge-verdict promoted";
            badge.innerText = "PROMOTED TO CHAMPION";
            title.innerText = `Challenger Promoted: New Champion ${report.new_active_version}`;
        } else {
            badge.className = "badge-verdict rejected";
            badge.innerText = "REJECTED (KEPT CHAMPION)";
            title.innerText = `Challenger Rejected: Active Champion Retained`;
        }
        reason.innerText = report.reason;

        // Reload Summary
        loadFeedbackSummary();

    } catch (err) {
        alert("Error executing adaptive retraining: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>⚡ Trigger Adaptive Retraining</span>";
    }
}
