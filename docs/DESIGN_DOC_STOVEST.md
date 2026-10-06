# Design Document: Stovest-Inspired Modern Minimalist Overhaul
## SmartFolio — Clean, Uncluttered Quantitative Investment Dashboard
**Inspired by [Stovest – Modern Minimalist Investment Dashboard by Asivest Fintech Agency for Delibix on Dribbble](https://dribbble.com/shots/26579883-Stovest-Modern-Minimalist-Investment-Dashboard)**

---

### 1. Executive Summary & Problem Analysis

#### The Problem with the Previous Iteration
The previous interface suffered from visual congestion and vertical stacking:
1. **Four Stacked Horizontal Bars at the Top:**
   - Bar 1: Bulky header (`padding: 1.25rem 2.5rem`) with large logo, multiple pills, status indicator, market tag, and legal button.
   - Bar 2: Intrusive full-width statutory warning notice with yellow badge and long paragraph.
   - Bar 3: Marquee ticker ribbon with 12 scrolling items.
   - Bar 4: Navigation tab bar with 5 large pill tabs.
   - *Result:* Over 300px of vertical screen real estate was consumed before the user could see any dashboard content.
2. **Oversized "Billboard" Action Buttons:**
   - Buttons (such as "Run Walk-Forward Simulation" and "Generate Optimized Allocation") spanned 100% of card width with glaring electric cyan backgrounds, screaming for attention and degrading visual sophistication.
3. **Competing Accent Colors & Visual Noise:**
   - Clashing yellow banners, neon greens, electric cyans, purples, and red badges fighting for optical hierarchy simultaneously.

#### The Stovest Solution
Asivest and Delibix’s **Stovest** design philosophy champions **radical visual decluttering, mobile/desktop minimalist elegance, and content-first clarity**:
- **Unified Single Header:** Consolidates branding, tab navigation, and live status into a single, slim, modern top bar (height: 60px).
- **Relocation of Non-Immediate Content:** The legal statutory caution is moved to a dignified, professional footer section, freeing the viewport for quantitative decision-making.
- **Balanced Component Proportions:** Action buttons are scaled to balanced, professional dimensions (`padding: 0.65rem 1.35rem`, inline-flex, aligned to card headers or natural form ends).
- **The "Positions" Control Center:** Data cards are clean, structured, and legible with subtle borders (`rgba(255, 255, 255, 0.07)`), generous breathing room, and restrained, high-contrast typography.

---

## 2. Color Palette & Typography Tokens (Stovest Design System)

```css
:root {
    /* Canvas & Surfaces */
    --sv-bg-canvas: #090c13;               /* Obsidian canvas */
    --sv-bg-surface: #111726;              /* Primary card surface */
    --sv-bg-surface-subtle: #0d121f;       /* Inset input/table surface */
    --sv-bg-surface-hover: #172033;        /* Interactive hover surface */
    
    /* Subtle Borders */
    --sv-border-subtle: rgba(255, 255, 255, 0.07);
    --sv-border-active: rgba(16, 185, 129, 0.4);
    --sv-border-focus: #10b981;

    /* Stovest Accent System */
    --sv-green-accent: #10b981;            /* Mint/Emerald growth & primary alpha */
    --sv-green-tint: rgba(16, 185, 129, 0.12);
    --sv-indigo-brand: #6366f1;            /* Primary brand & focus */
    --sv-indigo-tint: rgba(99, 102, 241, 0.12);
    --sv-red-loss: #f43f5e;                /* Risk & negative variance */
    --sv-amber-caution: #f59e0b;          /* Warnings & conservative profiles */

    /* Typography */
    --sv-text-primary: #f8fafc;            /* Crisp white figures & titles */
    --sv-text-secondary: #94a3b8;          /* High-contrast readable slate labels */
    --sv-text-muted: #64748b;              /* Quiet subtext and metadata */

    /* Radii & Elevation */
    --sv-radius-card: 14px;
    --sv-radius-control: 8px;
    --sv-radius-pill: 9999px;
    --sv-shadow-card: 0 4px 20px rgba(0, 0, 0, 0.35);
}
```

---

## 3. Layout Architecture: Before vs. After

### Before (Cluttered & Stacked)
```
[ Top Header with 4 Pills, Flags, Large Badges ]
[ Full-Width Yellow Statutory Banner & Paragraph ]
[ Full-Width Scrolling Marquee Ticker Ribbon ]
[ Full-Width Navigation Pill Tab Bar ]
[ Main Dashboard Content - Giant 100% Width Billboard Buttons ]
```

### After (Stovest Minimalist Architecture)
```
[ Unified Slim Header (60px): Logo | Integrated Segmented Nav Tabs | Status Pill ]
[ Main Dashboard: Clean 2-Column Grid with Ample Whitespace ]
  ├── Left: Compact Clean Configuration Panel with Inline Action Button
  └── Right: Positions & Allocation Control Center (Hero KPI, Donut, Breakdown)
[ Subtle, Clean Footer with Regulatory Disclosures & DPDP Compliance ]
```

---

## 4. Key Component Redesigns

1. **Unified Header (`.app-header`):**
   - Height reduced from 90px to a crisp 60px.
   - Navigation tabs integrated directly into the header center as a clean segmented pill control.
   - Status badge and legal link placed quietly on the right without screaming colors.
2. **Elimination of Visual Redundancy:**
   - Statutory notice moved from top banner to dedicated footer caution card.
   - Scrolling marquee removed from primary vertical flow; asset data displayed directly in the interactive Market Explorer and Universe Selector.
3. **Buttons Re-proportioned:**
   - Buttons are now appropriately sized (`width: auto; padding: 0.65rem 1.4rem`), featuring a refined emerald/indigo finish and subtle hover lift.
4. **Stock Universe Selector:**
   - Compact, clean asset cards with clear ticker name, sector tag, and crisp green selection indicator.
5. **Chart Containers:**
   - Transparent backdrops, ultra-light grid lines (`rgba(255, 255, 255, 0.04)`), and uncluttered legends.

---

*Approved for implementation across SmartFolio.*
