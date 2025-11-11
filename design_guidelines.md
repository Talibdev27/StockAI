# Stock Trading Prediction Platform - Design Guidelines

## Design Approach
**Reference-Based Approach**: Draw inspiration from **TradingView** and **Bloomberg Terminal** for financial data visualization, combined with modern web application patterns from platforms like **Robinhood** for approachability. This creates a professional, trustworthy interface that balances sophisticated financial tools with user-friendly design.

## Core Design Principles
1. **Data Clarity First**: All charts, metrics, and indicators must be immediately readable
2. **Professional Trust**: Visual design communicates reliability and expertise
3. **Dark Mode Priority**: Default to dark mode with optional light theme
4. **Information Density**: Maximize valuable data without overwhelming users

---

## Typography System

**Font Families**:
- Primary: Inter (for UI elements, labels, and body text)
- Secondary: JetBrains Mono or IBM Plex Mono (for numerical data, prices, percentages)
- Headlines: Poppins or Inter (weights: 600-700)

**Hierarchy**:
- Page Titles: 2xl-3xl, font-weight 700
- Section Headers: xl-2xl, font-weight 600
- Card Titles: lg-xl, font-weight 600
- Body Text: base, font-weight 400
- Data Labels: sm, font-weight 500
- Numerical Data: base-lg, monospace font, font-weight 600
- Small Captions/Metadata: xs-sm, font-weight 400

---

## Layout System

**Spacing Primitives**: Use Tailwind units of **2, 4, 6, 8, 12, 16** for consistent rhythm
- Component padding: p-4, p-6
- Section spacing: gap-6, gap-8
- Card padding: p-6, p-8
- Tight groupings: gap-2, gap-4

**Grid Structure**:
- Main dashboard: Multi-column grid layout
  - Desktop (lg:): 3-column grid for metric cards, 2-column for main chart + sidebar
  - Tablet (md:): 2-column for cards, stack chart and sidebar
  - Mobile: Single column, full-width components

**Container Strategy**:
- Full-width dashboard: max-w-7xl with appropriate padding
- Card containers: bg panels with rounded corners (rounded-lg or rounded-xl)
- Chart containers: Full-width within card, min-height of 400-500px for primary charts

---

## Component Library

### 1. Navigation Header
- Fixed top position with backdrop blur
- Logo/branding left-aligned
- Navigation menu center or left (Home, Predictions, Backtest Results, About)
- Right section: Stock search bar, theme toggle, user avatar/menu
- Height: h-16 with subtle bottom border

### 2. Stock Search & Selection
- Autocomplete dropdown with search input
- Popular stock chips displayed prominently (AAPL, GOOGL, MSFT, TSLA, AMZN)
- Selected stock display: Large stock symbol + company name + current price
- Quick-access timeframe selector: 1H, 4H, 1D, 1W, 1M buttons (pill-shaped, active state highlighted)

### 3. Main Prediction Chart Card
- Large, prominent card taking 60-70% of main content area
- Integrated timeframe controls at top
- Chart types toggle: Candlestick, Line, Area
- TradingView-style interactive chart showing:
  - Historical data (30 days) in neutral blue
  - Predicted future prices (5 days) in green (bullish) or red (bearish)
  - Grid lines, axis labels, tooltips on hover
- Volume bars at bottom of chart
- Min height: 500px on desktop, 350px on mobile

### 4. Prediction Summary Card
Compact card displaying key metrics in grid layout:
- Current Price (large, prominent)
- Predicted Price (next day, emphasized)
- Change percentage with color: green for positive, red for negative
- Prediction Confidence: percentage with progress bar
- Signal Indicator: Large badge/chip with BUY (green), HOLD (yellow/amber), SELL (red)

### 5. Technical Indicators Grid
6-card grid (3 columns on desktop, 2 on tablet, 1 on mobile):
- Each card shows one indicator:
  - RSI with circular gauge (0-100 scale)
  - MACD with mini line chart or values
  - Moving Averages (50-day, 200-day) with comparison
  - Bollinger Bands visualization
  - Volume with bar chart
  - Additional indicator (Stochastic, ATR, etc.)
- Cards sized uniformly with icon, label, value, and mini visualization

### 6. Model Performance Metrics
4 stat cards in horizontal row:
- Each card: Icon, metric name, large value, trend indicator
- Metrics: Prediction Accuracy (%), Sharpe Ratio, Win Rate (%), Total Return (%)
- Use subtle background gradients for visual appeal

### 7. Backtest Results Section
- Equity Curve Chart: Line chart showing portfolio growth over time
- Performance Metrics Table: Structured data display
  - Total Return, Max Drawdown, Number of Trades, Avg Trade Duration
- Monthly Returns Heatmap: Calendar-style grid with color intensity

### 8. Watchlist Sidebar
- Compact list of saved stocks
- Each item: Symbol, current price, change percentage, mini sparkline
- Add/remove buttons with icons
- Scrollable if list exceeds viewport

### 9. Risk Disclaimer Banner
- Sticky bottom banner or modal on first visit
- Amber/yellow background for attention
- Clear, bold text: "Educational purposes only. Not financial advice. Past performance doesn't guarantee future results."

---

## Data Visualization Standards

**Chart Colors** (Dark Mode):
- Historical prices: Blue (#3B82F6)
- Bullish predictions: Green (#10B981)
- Bearish predictions: Red (#EF4444)
- Grid lines: Gray (#374151, subtle)
- Text/labels: Light gray (#D1D5DB)

**Chart Colors** (Light Mode):
- Historical prices: Deep Blue (#1E40AF)
- Bullish predictions: Forest Green (#059669)
- Bearish predictions: Crimson Red (#DC2626)
- Grid lines: Light gray (#E5E7EB)
- Text/labels: Dark gray (#374151)

**Interactive Elements**:
- Hover states: Increase opacity, show tooltips
- Active timeframe: Solid background, bold text
- Chart crosshair: Thin lines following cursor

---

## Responsive Behavior

**Desktop (lg: 1024px+)**:
- Multi-column dashboard layout
- Full chart controls visible
- Sidebar watchlist alongside main content

**Tablet (md: 768px-1024px)**:
- 2-column card grids
- Chart maintains prominence
- Condensed navigation

**Mobile (base: <768px)**:
- Single column, stacked layout
- Collapsible watchlist (drawer or accordion)
- Simplified chart controls (dropdown instead of buttons)
- Touch-optimized chart interactions

---

## Special Considerations

**Loading States**: Skeleton screens with pulsing animation for charts and data cards
**Error States**: Empty state illustrations with retry buttons for failed data loads
**Real-time Updates**: Subtle pulse or highlight animation when prices update
**Accessibility**: High contrast in dark mode, ARIA labels for charts, keyboard navigation support

---

## Images
No hero images needed. This is a data-focused application where charts and real-time data are the primary visual elements. Any branding graphics should be minimal and logo-based.