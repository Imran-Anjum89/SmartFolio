# Design Document: StockWise-Inspired UI/UX Redesign
## SmartFolio — Next-Generation Intelligent Quantitative Investment Platform
**Design System, Aesthetic Specifications, and Component Architecture Inspired by Aryo Romadhon's StockWise (Hatypo Studio on Dribbble)**

---

### Document Overview
- **Document Title:** SmartFolio StockWise Design Specification Document
- **Design Inspiration:** [StockWise - Stock Market App by Aryo Romadhon for Hatypo Studio (Dribbble)](https://dribbble.com/shots/20774174--StockWise-Stock-Market-App)
- **Version:** v1.0.0
- **Status:** Approved for Implementation
- **Target Platform:** Desktop, Tablet, and Mobile Web (Responsive Single-Page Application)
- **Primary Design Paradigm:** Modern Fintech Glassmorphism, Deep Obsidian Dark Mode, Electric Neon Green Accents, Card-Driven Layouts, and Information-Dense Minimalist Visual Hierarchy.

---

## 1. Visual Analysis & Inspiration: The StockWise Aesthetic

### 1.1 Core Aesthetic Foundations
Aryo Romadhon and Hatypo Studio's **StockWise** stands as a celebrated hallmark of modern financial technology design on Dribbble. Unlike traditional legacy trading software cluttered with dense, unformatted tables and harsh terminal colors, StockWise introduces an elevated, human-centric approach to quantitative asset tracking:

1. **Deep Obsidian Canvas (`#090D15` / `#0D111A`):**
   - High-contrast, fatigue-free dark theme engineered for extended analytical sessions.
   - Soft, ambient multi-point radial glows (Cyan, Violet, Emerald) that add physical depth without distracting from critical financial data.
2. **Layered Glassmorphic Card Containers:**
   - Multi-depth cards with `backdrop-filter: blur(24px)` and subtle, crisp frosted borders (`1px solid rgba(255, 255, 255, 0.08)` to `0.14`).
   - Cards float gracefully above the background with soft ambient shadows (`0 14px 40px rgba(0, 0, 0, 0.45)`).
3. **Electric Mint / Neon Green Primary Accent (`#00F090` / `#10E77F`):**
   - Symbolic of financial growth, alpha, and operational vitality.
   - Used strategically for primary call-to-action buttons, key return metrics, positive deltas, and active status indicators.
4. **Information Hierarchy & Clean Component Modularity:**
   - **Hero Portfolio Balance Card:** High-impact monetary figures with percentage gain pills, micro trendlines, and risk factor badges.
   - **Visual Asset Cards / Chips:** Ticker symbols paired with geometric company monogram avatars, sector badges, and glowing selection rings.
   - **Segmented Control Pill Tabs:** Floating navigation with sleek pill containers and energetic glow indicators.
   - **Data-Dense Modern Plotly Charts:** Minimalist dark canvas charts with transparent backgrounds, soft grid lines (`rgba(255, 255, 255, 0.05)`), neon color palettes, and custom tooltips.

---

## 2. Design System Tokens & Color Palette

### 2.1 Color Tokens Specification

```css
:root {
    /* Base Obsidian Canvas */
    --sw-bg-base: #080C14;
    --sw-bg-surface: rgba(16, 22, 36, 0.78);
    --sw-bg-surface-elevated: rgba(22, 30, 48, 0.88);
    --sw-bg-card-hover: rgba(28, 38, 60, 0.92);
    --sw-bg-input: rgba(10, 14, 24, 0.85);

    /* Glassmorphic Borders */
    --sw-border-subtle: rgba(255, 255, 255, 0.07);
    --sw-border-card: rgba(255, 255, 255, 0.11);
    --sw-border-focus: rgba(0, 240, 144, 0.55);
    --sw-border-accent-cyan: rgba(56, 189, 248, 0.45);

    /* Signature StockWise Fintech Accents */
    --sw-neon-green: #00F090;           /* Primary Growth & Action Accent */
    --sw-neon-green-glow: rgba(0, 240, 144, 0.35);
    --sw-electric-cyan: #00E5FF;        /* Innovation & Tech Indicator */
    --sw-electric-cyan-glow: rgba(0, 229, 255, 0.35);
    --sw-iris-violet: #8B5CF6;          /* Machine Learning & Retraining */
    --sw-coral-red: #FF4D6D;            /* Risk, Volatility & Shortfalls */
    --sw-amber-gold: #FFB800;           /* Warnings & Conservative Risk */

    /* Typography Contrast (WCAG 2.1 AA/AAA) */
    --sw-text-white: #FFFFFF;           /* Primary Figures & Headers */
    --sw-text-primary: #F1F5F9;         /* Body Text & High-Readability Data */
    --sw-text-secondary: #94A3B8;       /* Meta Descriptions & Secondary Labels */
    --sw-text-muted: #64748B;           /* Subtext & Captions */

    /* Gradients */
    --sw-gradient-primary: linear-gradient(135deg, #00F090 0%, #00E5FF 100%);
    --sw-gradient-violet: linear-gradient(135deg, #8B5CF6 0%, #D946EF 100%);
    --sw-gradient-card-header: linear-gradient(180deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0) 100%);
    --sw-gradient-glow: radial-gradient(circle at 50% 0%, rgba(0, 240, 144, 0.12) 0%, transparent 60%);

    /* Radii & Elevation */
    --sw-radius-xl: 24px;
    --sw-radius-lg: 18px;
    --sw-radius-md: 12px;
    --sw-radius-sm: 8px;
    --sw-radius-pill: 9999px;

    /* Drop Shadows */
    --sw-shadow-card: 0 14px 38px 0 rgba(0, 0, 0, 0.48);
    --sw-shadow-button: 0 6px 24px 0 rgba(0, 240, 144, 0.32);
    --sw-shadow-glow-cyan: 0 6px 24px 0 rgba(0, 229, 255, 0.28);
}
```

### 2.2 Typography Hierarchy
- **Primary Typeface:** `Plus Jakarta Sans`, modern geometric sans-serif offering supreme clarity on digital screens.
- **Data & Numerical Typeface:** `JetBrains Mono` or tabular numbers (`font-variant-numeric: tabular-nums`) for currency values, percentage deltas, timestamps, and model coefficients.

| Style Level | Font Family | Size | Weight | Tracking | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Balance Metric** | Plus Jakarta Sans | 38px (2.4rem) | 800 Bold | -0.03em | Total Capital, Expected Return Hero Card |
| **Section Title (H2)** | Plus Jakarta Sans | 22px (1.38rem) | 700 SemiBold | -0.02em | Card Titles, Module Headers |
| **Card Subtitle (H3)** | Plus Jakarta Sans | 16px (1.0rem) | 600 SemiBold | -0.01em | Table Headers, Section Subtitles |
| **Body Primary** | Plus Jakarta Sans | 14px (0.88rem) | 400 Regular | Normal | Descriptive Explanations, Form Labels |
| **Tabular Numbers** | JetBrains Mono | 13px–15px | 500 Medium | 0.01em | Stock Tickers, Returns, Weights, Errors |
| **Micro Badge / Pill** | Plus Jakarta Sans | 11px (0.69rem) | 700 Bold | 0.06em | Status Indicators, Sector Tags, Deltas |

---

## 3. Component Architecture & UI Specifications

### 3.1 Live Market Ticker Ribbon (Top Bar)
- **Concept:** A StockWise-signature horizontal marquee ticker positioned right above the main navigation, displaying real-time NIFTY 50 benchmark prices and key stock momentum.
- **Visuals:** Dark translucent glass strip with auto-scrolling ticker badges (`NIFTY 50 24,850 ▲ +0.82%`, `RELIANCE ₹2,980 ▲ +0.74%`, `TCS ₹3,920 ▲ +1.15%`, `HDFCBANK ₹1,680 ▲ +0.45%`, `INFY ₹1,870 ▲ +1.32%`).
- **Interactive State:** Hovering pauses the ticker; clicking an asset scrolls to or activates the Market Explorer.

### 3.2 Segmented Control Pill Navigation
- **Concept:** Floating pill container with smooth rounded tabs replacing traditional square buttons.
- **Visuals:** Encased in a frosted capsule container (`background: rgba(16, 22, 36, 0.85); border-radius: 9999px; padding: 6px; border: 1px solid rgba(255, 255, 255, 0.08)`).
- **Active State:** Active tab receives a vibrant pill highlight with electric cyan/mint glow and white text (`background: rgba(0, 240, 144, 0.16); border: 1px solid rgba(0, 240, 144, 0.4); color: #00F090`).

### 3.3 StockWise Hero Portfolio Overview Card (Optimizer Output)
- **Concept:** The crown jewel of the StockWise design: an expansive hero financial card featuring:
  1. **Primary Capital & Return Header:** Large bold figures (`₹1,00,000` capital allocated, `+18.42%` expected return pill with green upward arrow).
  2. **Three-Pill Key Factor Ribbon:** Volatility Pill (`12.3% Ledoit-Wolf Risk`), Sharpe Score Pill (`1.52 Sharpe Ratio`), and Active Model Pill (`Ensemble v1.0`).
  3. **Visual Donut Allocation Chart:** A centered, thin-stroke donut chart with Sharpe ratio in the inner ring and glowing hover slices.
  4. **Actionable Explainability Drawer:** Rule-based decision tags per stock with quick modal triggers.

### 3.4 Visual Asset Selection Cards (Watchlist / Universe)
- **Concept:** Replacing plain text chips with modern StockWise asset cards.
- **Visuals:**
  - Distinct company monogram badge (e.g., `TC` for TCS in cyan, `RI` for Reliance in emerald, `HD` for HDFC in violet).
  - Ticker name bolded on top, company sector underneath (`IT`, `Banking`, `Energy`, `FMCG`).
  - Active selection receives an electric mint border (`border-color: #00F090; box-shadow: 0 0 16px rgba(0, 240, 144, 0.25)`) and checkmark badge.

### 3.5 Investment Configuration Panel
- **Form Controls:**
  - Capital input with prominent currency prefix badge in dark glass wrapper.
  - Risk Profile Radio Cards: 3 clean vertical cards with radio rings, lambda coefficients (`λ = 5.0`, `λ = 2.0`, `λ = 0.5`), and visual risk level indicator bars.
  - Custom select dropdowns with chevron icons and dark glass menus.
  - **StockWise Primary CTA Button:** High-energy button with electric mint-to-cyan gradient, drop shadow, and smooth micro-hover lift.

### 3.6 Data Visualization & Plotly Theming
- **Chart Palette:**
  - `#00F090` (StockWise Mint - Primary asset line / top allocation)
  - `#00E5FF` (Electric Cyan - Second allocation / moving average)
  - `#8B5CF6` (Iris Violet - Deep Learning LSTM / Ensemble)
  - `#38BDF8` (Sky Blue - Benchmark Equal-Weight)
  - `#FFB800` (Amber Gold - Traditional Markowitz baseline)
  - `#FF4D6D` (Coral Red - Drawdowns / negative residual errors)
- **Layout Tokens:**
  - Paper Background: `transparent`
  - Plot Background: `transparent`
  - Grid lines: `rgba(255, 255, 255, 0.04)`
  - Zero line: `rgba(255, 255, 255, 0.12)`
  - Fonts: `Plus Jakarta Sans`, color: `#94A3B8`

---

## 4. Micro-Interactions & Animation Guidelines

1. **Card Hover Dynamics:**
   - On cursor hover, glass cards elevate smoothly:
     `transform: translateY(-2px); transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);`
     `box-shadow: 0 18px 42px rgba(0, 0, 0, 0.55), 0 0 1px rgba(255, 255, 255, 0.15);`
2. **Button Active & Busy States:**
   - When calculating allocations or retraining:
     Buttons display animated spinner with `aria-busy="true"` and soft breathing glow animation.
3. **Pill Selection Ripple:**
   - Selecting a stock chip triggers an instantaneous radial ripple and border illumination.
4. **Modal Dialog Transitions:**
   - Modals enter with a backdrop blur increase (`backdrop-filter: blur(12px)`) and subtle scale-up (`scale(0.97)` to `scale(1)` with `opacity: 0` to `1` in `200ms`).

---

## 5. Accessibility & Compliance Verification

- **Color Contrast:** All numerical values, status badges, and interactive labels strictly pass WCAG 2.1 Level AA (minimum 4.5:1 ratio against dark obsidian backgrounds).
- **Keyboard Navigation:** High-contrast focus rings (`outline: 2px solid #00F090; outline-offset: 3px`) on all interactive controls.
- **Screen Reader Support:** Semantic ARIA roles, live regions for optimization calculation results, and clear label associations.

---

*Design Document Approved for SmartFolio Quantitative Platform UI Redesign.*
