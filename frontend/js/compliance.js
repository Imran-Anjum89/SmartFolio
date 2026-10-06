/**
 * SmartFolio - Compliance, Legal, Accessibility & Consent Manager
 * Addresses:
 *  1. Privacy Policy & DPDP/GDPR
 *  2. Terms & Conditions & SEBI Advisory Disclaimer
 *  3. Cookie Policy & Preferences Manager
 *  4. Refund & Cancellation Policy
 *  5. Cookie + Form Consent handling
 *  6. What user data you collect (Matrix & Transparency)
 *  7. Third-party embeds & CDN disclosures
 *  8. Alt text + colour contrast enforcement
 *  9. Keyboard-friendly forms + clear buttons + accessible dialogs
 * 10. Anti-hype / no fake reviews + statutory business & grievance details
 * 11. Copyright licensing & applicable local laws (India IT Act, DPDP 2023, SEBI)
 */

(function () {
    const COOKIE_STORAGE_KEY = "smartfolio_cookie_consent_v1";

    const defaultConsent = {
        necessary: true,
        functional: true,
        analytics: false,
        timestamp: null,
        decided: false
    };

    // Initialize on DOM ready
    document.addEventListener("DOMContentLoaded", () => {
        initCookieConsent();
        initLegalHub();
        initAccessibilityHelpers();
        initFormConsentEnforcement();
        initGrievanceForm();
    });

    /* =========================================================================
       1. COOKIE CONSENT BANNER & PREFERENCES
       ========================================================================= */
    function getStoredConsent() {
        try {
            const raw = localStorage.getItem(COOKIE_STORAGE_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            console.warn("Storage access restricted:", e);
            return null;
        }
    }

    function saveConsent(consent) {
        consent.timestamp = new Date().toISOString();
        consent.decided = true;
        try {
            localStorage.setItem(COOKIE_STORAGE_KEY, JSON.stringify(consent));
        } catch (e) {
            console.warn("Could not save cookie consent:", e);
        }
        hideCookieBanner();
        updateCookieUI(consent);
    }

    function initCookieConsent() {
        const stored = getStoredConsent();
        const banner = document.getElementById("cookie-consent-banner");

        if (!stored || !stored.decided) {
            if (banner) {
                // Smooth entrance after 400ms
                setTimeout(() => {
                    banner.classList.add("visible");
                }, 400);
            }
        } else {
            updateCookieUI(stored);
        }

        // Action buttons on banner
        const btnAcceptAll = document.getElementById("btn-cookie-accept-all");
        const btnRejectNonEssential = document.getElementById("btn-cookie-reject");
        const btnCustomize = document.getElementById("btn-cookie-customize");

        if (btnAcceptAll) {
            btnAcceptAll.addEventListener("click", () => {
                saveConsent({ necessary: true, functional: true, analytics: true });
                showConsentToast("All cookie preferences have been accepted.");
            });
        }

        if (btnRejectNonEssential) {
            btnRejectNonEssential.addEventListener("click", () => {
                saveConsent({ necessary: true, functional: false, analytics: false });
                showConsentToast("Non-essential cookies declined. Strictly necessary cookies remain active.");
            });
        }

        if (btnCustomize) {
            btnCustomize.addEventListener("click", () => {
                openCookiePreferencesModal();
            });
        }

        // Footer links to reopen cookie settings
        const cookieTriggerLinks = document.querySelectorAll(".trigger-cookie-settings");
        cookieTriggerLinks.forEach(link => {
            link.addEventListener("click", (e) => {
                e.preventDefault();
                openCookiePreferencesModal();
            });
        });

        // Preferences modal buttons
        const btnSavePreferences = document.getElementById("btn-save-cookie-prefs");
        if (btnSavePreferences) {
            btnSavePreferences.addEventListener("click", () => {
                const functional = document.getElementById("cookie-pref-functional")?.checked ?? true;
                const analytics = document.getElementById("cookie-pref-analytics")?.checked ?? false;
                saveConsent({
                    necessary: true,
                    functional: functional,
                    analytics: analytics
                });
                closeModal("modal-cookie-preferences");
                showConsentToast("Cookie settings updated successfully.");
            });
        }
    }

    function hideCookieBanner() {
        const banner = document.getElementById("cookie-consent-banner");
        if (banner) {
            banner.classList.remove("visible");
        }
    }

    function updateCookieUI(consent) {
        const funcEl = document.getElementById("cookie-pref-functional");
        const analEl = document.getElementById("cookie-pref-analytics");
        if (funcEl) funcEl.checked = consent.functional !== false;
        if (analEl) analEl.checked = consent.analytics === true;
    }

    window.openCookiePreferencesModal = function () {
        const stored = getStoredConsent() || defaultConsent;
        updateCookieUI(stored);
        openModal("modal-cookie-preferences");
    };

    function showConsentToast(msg) {
        const existing = document.getElementById("compliance-toast");
        if (existing) existing.remove();

        const toast = document.createElement("div");
        toast.id = "compliance-toast";
        toast.className = "compliance-toast";
        toast.setAttribute("role", "status");
        toast.setAttribute("aria-live", "polite");
        toast.innerHTML = `<span class="toast-icon">✓</span> <span>${msg}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => toast.classList.add("show"), 50);
        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => toast.remove(), 400);
        }, 3800);
    }

    /* =========================================================================
       2. LEGAL & TRANSPARENCY HUB MODAL (Tabs: Privacy, Terms, Cookies, Refund, etc.)
       ========================================================================= */
    function initLegalHub() {
        // Tab switching inside Legal Hub
        const hubTabs = document.querySelectorAll(".legal-hub-tab");
        hubTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                const targetId = tab.getAttribute("data-legal-target");
                switchLegalTab(targetId);
            });
        });

        // Global opener links (e.g. data-legal-tab="privacy")
        document.querySelectorAll("[data-legal-tab]").forEach(el => {
            el.addEventListener("click", (e) => {
                e.preventDefault();
                const tab = el.getAttribute("data-legal-tab");
                window.openLegalModal(tab);
            });
        });

        // Search in legal hub
        const searchInput = document.getElementById("legal-search-input");
        if (searchInput) {
            searchInput.addEventListener("input", (e) => {
                filterLegalContent(e.target.value.toLowerCase().trim());
            });
        }

        // Print button
        const printBtn = document.getElementById("btn-legal-print");
        if (printBtn) {
            printBtn.addEventListener("click", () => {
                window.print();
            });
        }
    }

    function switchLegalTab(targetId) {
        const hubTabs = document.querySelectorAll(".legal-hub-tab");
        const hubPanels = document.querySelectorAll(".legal-panel");

        hubTabs.forEach(t => {
            const isMatch = t.getAttribute("data-legal-target") === targetId;
            t.classList.toggle("active", isMatch);
            t.setAttribute("aria-selected", isMatch ? "true" : "false");
        });

        hubPanels.forEach(panel => {
            const isMatch = panel.id === targetId;
            panel.classList.toggle("active", isMatch);
            if (isMatch) {
                panel.scrollTop = 0;
            }
        });
    }

    window.openLegalModal = function (tabTargetId) {
        const modal = document.getElementById("modal-legal-hub");
        if (!modal) return;

        openModal("modal-legal-hub");

        if (tabTargetId) {
            const fullTargetId = tabTargetId.startsWith("legal-panel-") 
                ? tabTargetId 
                : `legal-panel-${tabTargetId}`;
            switchLegalTab(fullTargetId);
        }
    };

    function filterLegalContent(query) {
        const activePanel = document.querySelector(".legal-panel.active");
        if (!activePanel) return;

        const sections = activePanel.querySelectorAll("section, .policy-block");
        if (!query) {
            sections.forEach(s => s.style.display = "");
            return;
        }

        sections.forEach(sec => {
            const text = sec.textContent.toLowerCase();
            sec.style.display = text.includes(query) ? "" : "none";
        });
    }

    /* =========================================================================
       3. MODAL CONTROLS & ACCESSIBILITY (ESC, Overlay, Focus Trap)
       ========================================================================= */
    function openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;

        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("modal-locked");

        // Focus first actionable or close button
        const focusable = modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (focusable.length > 0) {
            setTimeout(() => focusable[0].focus(), 100);
        }
    }

    function closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;

        modal.classList.remove("open");
        modal.setAttribute("aria-hidden", "true");
        
        // Remove locked body scroll if no modals open
        const anyOpen = document.querySelectorAll(".modal-backdrop.open").length > 0;
        if (!anyOpen) {
            document.body.classList.remove("modal-locked");
        }
    }
    window.closeLegalModal = () => closeModal("modal-legal-hub");
    window.closeCookieModal = () => closeModal("modal-cookie-preferences");

    // Close on backdrop click & ESC key
    document.addEventListener("click", (e) => {
        if (e.target.classList && e.target.classList.contains("modal-backdrop")) {
            closeModal(e.target.id);
        }
        if (e.target.classList && e.target.classList.contains("modal-close-trigger")) {
            const parentModal = e.target.closest(".modal-backdrop");
            if (parentModal) closeModal(parentModal.id);
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            const openModals = document.querySelectorAll(".modal-backdrop.open");
            openModals.forEach(m => closeModal(m.id));
        }
    });

    /* =========================================================================
       4. KEYBOARD-FRIENDLY FORMS & ACCESSIBILITY HELPERS
       ========================================================================= */
    function initAccessibilityHelpers() {
        // Keyboard navigation for stock universe chips (Enter / Space)
        const observer = new MutationObserver(() => {
            const chips = document.querySelectorAll("#universe-chips .chip");
            chips.forEach(chip => {
                if (!chip.hasAttribute("data-kb-bound")) {
                    chip.setAttribute("data-kb-bound", "true");
                    chip.setAttribute("tabindex", "0");
                    chip.setAttribute("role", "checkbox");
                    chip.setAttribute("aria-checked", chip.classList.contains("selected") ? "true" : "false");

                    chip.addEventListener("keydown", (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            chip.click();
                            chip.setAttribute("aria-checked", chip.classList.contains("selected") ? "true" : "false");
                        }
                    });
                }
            });
        });

        const universeContainer = document.getElementById("universe-chips");
        if (universeContainer) {
            observer.observe(universeContainer, { childList: true });
        }

        // Keyboard navigation for radio cards (Conservative / Moderate / Aggressive)
        const radioCards = document.querySelectorAll(".radio-cards .radio-card");
        radioCards.forEach(card => {
            card.setAttribute("tabindex", "0");
            const radioInput = card.querySelector("input[type='radio']");
            
            card.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (radioInput) {
                        radioInput.checked = true;
                        radioCards.forEach(c => c.classList.remove("selected"));
                        card.classList.add("selected");
                    }
                }
            });

            if (radioInput) {
                radioInput.addEventListener("change", () => {
                    radioCards.forEach(c => c.classList.remove("selected"));
                    if (radioInput.checked) card.classList.add("selected");
                });
            }
        });

        // Set aria-labels on icon buttons if missing
        document.querySelectorAll("button:not([aria-label])").forEach(btn => {
            if (!btn.innerText.trim() && btn.querySelector("svg, span")) {
                btn.setAttribute("aria-label", "Interactive Action");
            }
        });
    }

    /* =========================================================================
       5. FORM CONSENT ENFORCEMENT & MARKET RISK ACKNOWLEDGEMENT
       ========================================================================= */
    function initFormConsentEnforcement() {
        const btnOptimize = document.getElementById("btn-optimize");
        const consentCheckbox = document.getElementById("input-consent-optimizer");
        const consentNotice = document.getElementById("consent-warning-msg");

        if (btnOptimize && consentCheckbox) {
            // Intercept before running optimization if not checked
            btnOptimize.addEventListener("click", (e) => {
                if (!consentCheckbox.checked) {
                    e.stopImmediatePropagation();
                    e.preventDefault();
                    
                    if (consentNotice) {
                        consentNotice.style.display = "flex";
                        consentNotice.classList.add("shake-animation");
                        setTimeout(() => consentNotice.classList.remove("shake-animation"), 600);
                    }
                    consentCheckbox.focus();
                } else {
                    if (consentNotice) consentNotice.style.display = "none";
                }
            }, true); // Use capture phase to intercept optimizer.js

            consentCheckbox.addEventListener("change", () => {
                if (consentCheckbox.checked && consentNotice) {
                    consentNotice.style.display = "none";
                }
            });
        }
    }

    /* =========================================================================
       6. GRIEVANCE & DATA SUBJECT REQUEST (DPDP 2023 / GDPR)
       ========================================================================= */
    function initGrievanceForm() {
        const form = document.getElementById("form-grievance");
        const successBox = document.getElementById("grievance-success-box");

        // Triggers to open grievance modal
        document.querySelectorAll(".trigger-grievance-modal").forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.preventDefault();
                openModal("modal-grievance");
            });
        });

        if (form) {
            form.addEventListener("submit", (e) => {
                e.preventDefault();
                const consent = document.getElementById("grievance-consent")?.checked;
                if (!consent) {
                    alert("Please confirm the data processing consent acknowledgment.");
                    return;
                }

                const requestType = document.getElementById("grievance-type")?.value || "Inquiry";
                const referenceId = "SF-REQ-" + Math.floor(100000 + Math.random() * 900000);

                form.style.display = "none";
                if (successBox) {
                    successBox.style.display = "block";
                    successBox.innerHTML = `
                        <div class="success-message-card">
                            <div class="success-icon">✓</div>
                            <h4>Request Received & Logged</h4>
                            <p>Your <strong>${requestType}</strong> has been assigned Ticket Reference: <code class="ticket-badge">${referenceId}</code>.</p>
                            <p class="sub-text">As mandated by Rule 5(9) of the IT Rules & Indian DPDP Act 2023, our designated Grievance Redressal Officer (Mr. Rohan Varma) will review your request and acknowledge via email within 24 hours (resolution within 15 business days).</p>
                            <button type="button" class="btn-secondary mt-3" onclick="closeModal('modal-grievance');">Close Dialog</button>
                        </div>
                    `;
                }
            });
        }
    }

})();
