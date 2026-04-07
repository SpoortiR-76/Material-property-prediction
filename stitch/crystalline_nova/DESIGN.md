# Design System: The Molecular Ethereal

## 1. Overview & Creative North Star
The Creative North Star for this design system is **"The Digital Laboratory."** 

In the context of Material Property Prediction, the UI must feel as precise as a scientific instrument yet as expansive as the molecular structures it analyzes. This system rejects the "flat, boxed" nature of traditional SaaS dashboards. Instead, it utilizes **Refractive Depth**—a philosophy where data floats in a pressurized, dark void, separated only by density and light. 

By leveraging intentional asymmetry and overlapping glass surfaces, we move away from "templates" toward an editorial experience. We treat ML predictions not just as data points, but as high-value insights that deserve a premium, high-contrast environment.

---

## 2. Colors & Surface Philosophy
The palette is rooted in a deep, atmospheric navy (`surface: #121222`), providing a high-contrast canvas for vibrant molecular accents.

### The Color Logic
*   **Primary (`#d0bcff`) & Secondary (`#4cd7f6`):** Used primarily for "Active Energy"—the path of the user’s intent.
*   **Tertiary (`#4edea3`):** Reserved for "Validation"—successful predictions and stable material states.
*   **Accent Orange (`#f59e0b`):** Used sparingly for "Anomalies" or high-heat material properties.

### The "No-Line" Rule
Standard 1px solid borders are strictly prohibited for structural sectioning. Boundaries must be defined through:
1.  **Tonal Shifts:** Placing a `surface-container-low` (`#1a1a2b`) element against the `surface` (`#121222`) background.
2.  **Refraction:** Using the Glassmorphism specification for floating modules.

### Glass & Gradient Implementation
To achieve "Visual Soul," do not use flat fills for primary actions. 
*   **Signature Gradient:** All high-level CTAs must utilize a linear gradient from `primary` (`#d0bcff`) to `secondary` (`#4cd7f6`) at a 135-degree angle.
*   **The Glass Formula:** Floating cards must use `background: rgba(255, 255, 255, 0.05)` with a `backdrop-filter: blur(20px)`. This allows the deep navy background to bleed through, creating a "frosted laboratory" aesthetic.

---

## 3. Typography: Editorial Precision
The system pairs **Space Grotesk** (Display/Headlines) with **Inter** (Body/Labels).

*   **Display & Headlines:** Use Space Grotesk to lean into the scientific, technical nature of the application. The wide apertures and geometric forms convey modern authority. Use `display-lg` (3.5rem) with negative letter-spacing (-0.02em) for hero headlines to create a bespoke, high-end feel.
*   **Body & Utility:** Inter is the workhorse. Its high x-height ensures readability of complex chemical formulas and ML confidence scores.
*   **Hierarchy as Identity:** Create "Typographic Tension" by pairing massive `display-sm` headlines next to tiny, all-caps `label-md` metadata. This contrast eliminates the "middle-ground" blandness found in generic UI.

---

## 4. Elevation & Depth
In this system, depth is a measure of "Molecular Density."

### The Layering Principle
Do not use shadows to create hierarchy; use **Stacking Tiers**.
1.  **Base Layer:** `surface` (#121222)
2.  **Sectional Layer:** `surface-container-low` (#1a1a2b)
3.  **Interactive Layer:** `surface-container-high` (#29283a)
4.  **Floating Glass Layer:** `rgba(255, 255, 255, 0.05)` + 20px blur.

### Ambient Shadows & Ghost Borders
*   **Ambient Shadows:** For floating modals, use a 40px blur shadow at 6% opacity using the `primary` token color. This mimics the glow of a backlit screen rather than a physical shadow.
*   **The Ghost Border:** If an element requires a container (like an input), use the `outline-variant` token at 15% opacity. This creates a "barely-there" structural hint.

---

## 5. Components

### Buttons (High-Energy Elements)
*   **Primary:** Gradient fill (`primary` to `secondary`), `rounded-md` (0.75rem). Text is `on-primary-fixed` (deep purple) for maximum legibility.
*   **Secondary (Glass):** Glass background, 10% opacity `white` border, `backdrop-blur`.
*   **Tertiary:** No background. `label-md` typography with a `primary` color glow on hover.

### Input Fields (The Data Portal)
*   **Style:** `surface-container-lowest` fill, `backdrop-filter: blur(10px)`.
*   **Focus State:** A 2px glow using the `primary` (`#d0bcff`) token. Avoid "boxing" the user in; keep the interaction feeling light and translucent.

### Selection Chips (Atomic Filters)
*   Used for selecting material elements (e.g., "Conductivity," "Elasticity").
*   **State:** Unselected chips are semi-transparent. Selected chips glow with the `secondary` (`#4cd7f6`) color.

### Prediction Cards & Lists
*   **No Dividers:** Forbid the use of horizontal lines. Use `spacing-8` (2rem) of vertical whitespace to separate items, or transition the background color slightly.
*   **Molecular Overlays:** Use `surface-container-highest` for "hover" states on list items to create a subtle "lift" effect.

---

## 6. Do’s and Don’ts

### Do
*   **Embrace Negative Space:** Allow large data visualizations to "breathe" by using `spacing-24` around major containers.
*   **Use Intentional Asymmetry:** Align the primary prediction value to the far right of a card while keeping metadata on the left to break the standard "centered" grid.
*   **Layer Glass:** Place a Glassmorphism chip inside a Glassmorphism card to create complex, high-end refractive depth.

### Don’t
*   **Don't use pure black (#000):** It kills the "atmospheric" navy feel. Always use the `surface` token.
*   **Don't use 100% opaque borders:** These act as visual "walls" that break the fluid, scientific aesthetic.
*   **Don't use standard drop shadows:** Avoid the "floating sticker" look. Use tonal shifts or ambient glows instead.
*   **Don't clutter with lines:** If you feel the need for a divider, increase the `spacing` scale instead.