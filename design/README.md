# Handoff: ClaimWise UI (Next.js implementation)

## Overview
ClaimWise is an AI insurance-claims assistant. The user files a claim by chatting in plain language with a robot mascot, **Wise**, who maps their words to the right forms, flags gaps, drafts the submission, and tracks the claim to payout. This package covers six screens: Landing, Chat (light + dark), the Wise mascot states, Empty/Error states, and Mobile (chat + slide-in history drawer).

## About the Design Files
The files in this bundle are **design references created in HTML** — a prototype showing the intended look and behavior. They are **not production code to copy directly**. Your task is to **recreate these designs in a Next.js app** using its own patterns (React components, CSS Modules / Tailwind / styled-components — whatever the project uses). If the Next.js project doesn't exist yet, scaffold one (`create-next-app`, App Router, TypeScript) and implement there.

`ClaimWise UI.dc.html` is a design-tool file. Ignore its custom tags (`<x-dc>`, `<helmet>`, `.dc.html` wrapper) — the **markup, inline styles, and CSS classes inside it are the source of truth** for structure and values. `reference/industry-styles.css` is the full design-system stylesheet all the class names (`.blueprint`, `.btn`, `.card`, `.tag`, `.nav`, `.input`, etc.) come from — port it (see Design Tokens).

## Fidelity
**High-fidelity.** Colors, typography, spacing, and layout are final. Recreate pixel-for-pixel. Exact hex values, fonts, and measurements are below.

## Design language (read first)
This is the **Industry** design system — a technical *wireframe / blueprint* aesthetic. Non-negotiable rules that make it look right:
- **Square corners everywhere.** `border-radius: 0` on cards, buttons, inputs, tags. The only rounded things are the Wise robot's body and chat bubbles from the user.
- **Line-drawing objects, not filled cards.** Cards/panels are transparent with a **1px hairline border** (`--color-divider`), never a fill + drop shadow.
- **Registration marks.** Every framed object (cards, hero image, chat bubbles from Wise, artboards) wears the `.blueprint` treatment: 1px border + four small `+` crosshair marks that sit *outside* the four corners. See `.blueprint`/`.corner` in the CSS — reproduce it as a `<Blueprint>` wrapper component.
- **One solid object on the board:** the primary button (solid steel accent fill). Everything else is outline/transparent.
- **Steel accent only.** No decorative color beyond the steel `#5980a6`. The single exception is a muted terracotta `#b46a4d` used *only* for error/failure states.

## Typography
Load from Google Fonts (use `next/font/google`).
- **Headings / UI labels / buttons:** `Barlow Condensed`, weight **600** (also 400). `letter-spacing: -0.015em`, `line-height: 1.12`.
- **Body:** `Barlow`, weights 400 / 500 / 700. Base `font-size: 15px`, `line-height: 1.55`.
- Heading scale (px): h1 42, h2 32, h3 25, h4 20, h5 16, h6 13 (h6 is uppercase, `letter-spacing: 0.08em`). The Landing hero uses a custom **58px** h1.
- Small labels / kickers: 10–11px, uppercase, `letter-spacing: 0.1–0.12em`, in the accent color.

```tsx
import { Barlow, Barlow_Condensed } from 'next/font/google';
export const barlow = Barlow({ subsets: ['latin'], weight: ['400','500','700'], variable: '--font-body' });
export const barlowCondensed = Barlow_Condensed({ subsets: ['latin'], weight: ['400','600'], variable: '--font-heading' });
```

## Design Tokens
Drop these into `:root` (light) and a `[data-theme="dark"]` / `.dark` scope. Full ramps are in `reference/industry-styles.css`.

```css
:root {
  --color-bg: #f2f2f3;        /* app ground */
  --color-surface: #e9e9ea;   /* bubbles, insets */
  --color-text: #1d1f20;
  --color-accent: #5980a6;    /* steel — primary */
  --color-accent-600: #597ea3;/* primary hover */
  --color-accent-700: #416180;/* primary active / mascot stroke */
  --color-accent-100: #eef6ff;/* accent tint fills */
  --color-accent-800: #2c455d;/* text on accent tint */
  --color-divider: color-mix(in srgb, #1d1f20 16%, transparent);
  --color-error: #b46a4d;     /* terracotta — errors only */
  --color-success: #3f7a53;   /* done state chip */

  --font-heading: "Barlow Condensed", system-ui, sans-serif;
  --font-body: "Barlow", system-ui, sans-serif;

  /* spacing scale (0.85× density) */
  --space-1: 3.4px; --space-2: 6.8px; --space-3: 10.2px;
  --space-4: 13.6px; --space-6: 20.4px; --space-8: 27.2px;
  --radius-sm: 2px; --radius-md: 4px; --radius-lg: 7px; /* but components override to 0 */

  --shadow-sm: 0 1px 2px color-mix(in srgb, #2b2b2d 14%, transparent);
  --shadow-md: 0 3px 10px color-mix(in srgb, #2b2b2d 16%, transparent);
  --shadow-lg: 0 12px 32px color-mix(in srgb, #2b2b2d 22%, transparent);
}

/* DARK MODE (used by Chat dark screen) */
[data-theme="dark"] {
  --color-bg: #131a24;
  --color-surface: #1b242f;   /* bubbles/panels */
  --color-sidebar: #0f151d;   /* sidebar bg */
  --color-text: #e7edf3;
  --color-accent: #749dc4;    /* brighter steel on dark */
  --color-mascot: #8fb2d6;    /* Wise stroke on dark */
  --color-divider: rgba(255,255,255,0.14);
}
```

Component rules (from `industry-styles.css` — port verbatim):
- `.btn` — `font-family: var(--font-heading)`, 14px, 1px border, `border-radius: 0`, padding `6.8px ~12px`.
  - `.btn-primary` → `background: var(--color-accent); color: var(--color-bg)`; hover `--color-accent-600`; active `--color-accent-700`.
  - `.btn-secondary` → transparent, 1px divider border; hover `color-mix(#1d1f20 7%)`.
- `.card`/panel → transparent, 1px divider border, square, `padding: var(--space-3)`.
- `.tag` → 11px, padding `3px 10px`. `.tag-accent` = `#eef6ff` bg / `#2c455d` text. `.tag-outline` = 1px accent border, accent text.
- `.input` → `background: var(--color-surface)`, 1px divider border, square, 36px min-height; focus border → accent.
- **Focus ring (all interactive):** `:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }` — never the browser default.
- `::selection` → `color-mix(in srgb, var(--color-accent) 30%, transparent)`.

### The `<Blueprint>` wrapper (crosshair frame)
```tsx
export function Blueprint({ children, className, style }: {...}) {
  return (
    <div className={`blueprint ${className ?? ''}`} style={{ position:'relative', border:'1px solid var(--color-divider)', borderRadius:0, ...style }}>
      {(['tl','tr','bl','br'] as const).map(c => <i key={c} className={`corner ${c}`} />)}
      {children}
    </div>
  );
}
```
`.corner` CSS (copy from `industry-styles.css`): 11×11px, drawn with two 1px pseudo-elements forming a `+`, positioned `-6px` outside each corner, color `color-mix(in srgb, var(--color-text) 55%, transparent)`.

## Screens / Views

### 1. Landing
- **Purpose:** marketing first-touch; convert to "Start a claim".
- **Layout:** full-width `.blueprint` frame containing: (a) top `nav` bar (brand + Wise icon left, links `Product / How it works / Pricing / Support`, then `Sign in` secondary btn + `Start a claim` primary btn), border-bottom hairline; (b) hero — 2-col grid `1.05fr .95fr`, `gap:40px`, `padding:56px 48px 40px`; left = `tag-outline` "AI claims assistant", 58px h1 (`File your insurance claim by just talking to Wise.`), 17px muted paragraph, two CTA buttons, a row of three "◇" feature stats; right = square `.blueprint.duotone` panel (aspect 5/4) with a large Wise mascot centered; (c) 3-col spec plate (border-top), each cell: big steel number (01/02/03), h4, 14px muted copy, 1px column dividers.
- **Copy:** step titles `Describe it / Wise assembles / Review & send`. Stats: `Avg. claim in 6 min · 24/7 · no queues · SOC-2 secure`.

### 2. Chat — light (core screen)
- **Purpose:** the conversation where claims get filed.
- **Layout:** `.blueprint` frame `1080×720`, 2-col grid `280px 1fr`.
- **Sidebar (280px, bg `#ededee`, right hairline):** brand row (Wise icon + "ClaimWise"); `+ New claim` **primary block button**; `RECENT` label; nav list of conversation items. **Active item:** `border-left: 2px solid #5980a6` + `background: color-mix(#5980a6 12%)`; each item = title + 11px muted status line (`Active · #CW-4821`, `Submitted · 2d ago`, `Paid out · Jun 3`, `Draft`). Footer pinned bottom: 28px square avatar initials `DA` + name + `⚙`.
- **Main:** header (claim title + `#CW-4821 · Homeowners`, right-aligned `.tag-accent` "◐ In progress · 60%"); scroll area `padding:26px 22px`, `gap:20px`:
  - **Wise (bot) message:** row = 30px Wise avatar + `.blueprint` bubble, `background:#e9e9ea`, `padding:14px 16px`, max-width 78%.
  - **User message:** right-aligned, `background:#5980a6`, `color:#f2f2f3`, `padding:14px 16px`, max-width 70%, **square corners**, no crosshairs.
  - **Auto-drafted summary card:** `.blueprint` `width:340px`; kicker `CLAIM SUMMARY — AUTO-DRAFTED`; rows (label left muted / value right) with hairline row dividers: `Incident date / Cause / Est. damage` (last value in accent "Needs photo").
- **Composer:** border-top; `.blueprint` inset (`#e9e9ea`) = 📎 + placeholder "Message Wise…" + primary `Send ↑` button. Disclaimer under it, centered, 11px muted: "Wise can make mistakes. Review your claim summary before submitting."

### 3. Chat — dark
Same layout, dark tokens. bg `#131a24`, sidebar `#0f151d`, bubbles/panels `#1b242f`, text `#e7edf3`, dividers `rgba(255,255,255,.14)`, accent `#749dc4`, Wise stroke `#8fb2d6`. Primary button on dark: `background:#749dc4; color:#0f151d`. User bubble: `#749dc4` bg / `#0f151d` text. **Includes the typing indicator** (see Interactions).

### 4. Wise — mascot states
`.blueprint` frame, 4 equal cells with column dividers. Each cell = a 120×130 Wise SVG + a state chip + a 12px caption. States: **Idle** (round eyes, smile — accent `#416180`), **Thinking** (small eyes looking up, flat mouth, two thought bubbles top-right), **Done** (`^ ^` happy-arc eyes, big smile, a check on the body — success chip `#3f7a53`), **Error** (`x x` eyes, wavy mouth — stroke terracotta `#b46a4d`, chip "! Something's off"). Use these as loading/success/failure feedback throughout.

### 5. Empty & Error
Two `520×440` `.blueprint` frames, centered content.
- **Empty:** idle Wise, h3 "Nothing here yet", muted copy, primary "Start your first claim →".
- **Error:** terracotta Wise, h3 "Submission didn't go through", muted copy about draft being saved, mono line `Error CW-503 · gateway timeout`, buttons `Try again` (primary) + `Save & exit` (secondary).

### 6. Mobile
Two `390×800` frames.
- **Chat:** faux status bar (34px), header with `☰` menu + claim title + Wise icon, message list (same bubble rules, tighter), composer with `↑` button.
- **Drawer open:** dimmed scrim (`color-mix(#1d1f20 40%)`) over the chat, sidebar slid in from left (`inset:0 70px 0 0`, `box-shadow: var(--shadow-lg)`) — same content as the desktop sidebar.
- **Responsive rule:** below ~768px the sidebar becomes this off-canvas drawer toggled by `☰`; desktop shows it as a fixed 280px column.

## Interactions & Behavior
- **Nav:** clicking a sidebar item loads that conversation and sets the active `border-left`/tint. `+ New claim` opens a fresh empty thread (Empty state).
- **Send:** posts a user bubble (right, solid steel), then Wise responds. While Wise is "working," show the **typing indicator** then swap in the message.
- **Typing indicator:** three 7px dots in a `.blueprint` bubble beside a Wise avatar whose mouth is a flat line. Each dot pulses: `@keyframes dotp { 0%,60%,100% { opacity:.25; transform:translateY(0) } 30% { opacity:1; transform:translateY(-3px) } }`, `1.4s ease-in-out infinite`, dot 2 delay `.2s`, dot 3 delay `.4s`.
- **Mascot state animation:** cross-fade / morph Wise between Idle → Thinking (on send) → Done (on submit success) or Error (on failure). Keep it subtle; ~200ms transitions.
- **Submit success:** show Done Wise + success chip. **Submit failure:** show Error screen (#5) + terracotta Wise.
- **Hover/active/focus:** exactly as the token rules above (accent-600 hover, accent-700 active, 2px accent focus-visible ring). Sidebar items get a subtle tint on hover.
- **Theme toggle:** flip `data-theme` on `<html>`; persist to `localStorage`; respect `prefers-color-scheme` on first load.

## State Management
- `conversations: Conversation[]` (id, title, status: 'draft'|'active'|'submitted'|'paid', messages).
- `activeConversationId`.
- `messages: {role: 'wise'|'user', content, card?}[]`.
- `wiseState: 'idle'|'thinking'|'done'|'error'` (drives mascot + loaders).
- `theme: 'light'|'dark'`.
- `drawerOpen: boolean` (mobile).
- `claimSummary` (structured fields Wise fills as the chat progresses).
- Data fetching: chat send → API (streaming assistant reply recommended); claim submit → insurer gateway (handle timeout → error state).

## Assets
- **Wise mascot:** pure inline SVG, no image files — see `reference/Wise.tsx` for a ready React component with all four `state` variants (`idle | thinking | done | error`) and a `stroke`/`size` prop. Reuse it everywhere (nav icon, avatars, hero, states) by changing `size`.
- **Icons:** the system uses **Lucide** at `stroke-width: 1.5`. Replace the placeholder glyphs (☰ 📎 ⚙ ↑ ◇ ◐ ✓) with Lucide (`Menu`, `Paperclip`, `Settings`, `ArrowUp`, `Diamond`, `CircleDashed`, `Check`). `npm i lucide-react`.
- **Fonts:** Barlow + Barlow Condensed via `next/font/google` (no files to ship).
- No photography in the mock (the hero panel shows Wise, framed with `.duotone`). If real photos are added later, wrap them in `.duotone` (desaturate + steel `mix-blend-mode: color` overlay).

## Files
- `ClaimWise UI.dc.html` — the full prototype (all six screens). Open in a browser to see every screen; view source for exact markup/values.
- `reference/industry-styles.css` — the complete Industry design-system stylesheet (tokens + all component classes). Port this first.
- `reference/Wise.tsx` — the mascot as a drop-in React/SVG component.
- `reference/tokens.css` — the token block above, ready to paste into `globals.css`.
