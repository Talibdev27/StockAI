import { useMemo, useState } from "react";
import StockSelector from "@/components/StockSelector";
import TimeframeSelector, { Timeframe } from "@/components/TimeframeSelector";
import PriceChart from "@/components/PriceChart";
import PredictionSummary from "@/components/PredictionSummary";
import TechnicalIndicators from "@/components/TechnicalIndicators";
import PerformanceMetrics from "@/components/PerformanceMetrics";
import Watchlist from "@/components/Watchlist";
import RiskDisclaimer from "@/components/RiskDisclaimer";
import HelpSection from "@/components/HelpSection";
import { Activity, BarChart3, TrendingUp, Move, Gauge, Target, Award, DollarSign, Loader2, AlertCircle, CheckCircle2, XCircle } from "lucide-react";
import { useHistorical, usePrediction, useQuote, useIndicators, useBackendHealth } from "@/hooks/useData";
import PredictionAccuracy from "@/components/PredictionAccuracy";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const stockPriceRanges: Record<string, { base: number; volatility: number }> = {
  AAPL: { base: 162, volatility: 5 },
  GOOGL: { base: 138, volatility: 4 },
  MSFT: { base: 378, volatility: 8 },
  TSLA: { base: 242, volatility: 12 },
  AMZN: { base: 148, volatility: 6 },
};

const generateChartData = (stock: string, timeframe: Timeframe) => {
  const { base, volatility } = stockPriceRanges[stock] || stockPriceRanges.AAPL;
  const points = timeframe === "1H" ? 24 : timeframe === "4H" ? 18 : timeframe === "1D" ? 30 : timeframe === "1W" ? 12 : 6;
  
  const data = [];
  let price = base;
  
  for (let i = 0; i < points; i++) {
    const change = (Math.random() - 0.5) * volatility;
    price = price + change;
    data.push({
      date: timeframe === "1H" ? `${i}:00` : timeframe === "4H" ? `Day ${i * 4}h` : `Day ${i + 1}`,
      price: parseFloat(price.toFixed(2)),
    });
  }
  
  for (let i = 0; i < 5; i++) {
    const change = (Math.random() - 0.3) * volatility;
    price = price + change;
    data.push({
      date: timeframe === "1H" ? `${points + i}:00` : `+${i + 1}`,
      price: parseFloat(price.toFixed(2)),
      predicted: true,
    });
  }
  
  return data;
};

export default function Dashboard() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [searchQuery, setSearchQuery] = useState("");
  const [timeframe, setTimeframe] = useState<Timeframe>("1D");

  // Map timeframe to API params
  const timeframeMap: Record<Timeframe, { range: string; interval: string }> = {
    "5m": { range: "60d", interval: "5m" },   // Yahoo max for 5m
    "15m": { range: "60d", interval: "15m" }, // Yahoo max for 15m
    "1H": { range: "730d", interval: "1h" },  // Yahoo max for 1h (~2y)
    "4H": { range: "5d", interval: "4h" },
    "1D": { range: "1y", interval: "1d" },
    "1W": { range: "2y", interval: "1wk" },
    "1M": { range: "5y", interval: "1mo" },
  };

  // Map timeframe to prediction horizon
  const horizonMap: Record<Timeframe, number> = {
    "5m": 60,   // Predict next 60 intervals (5 hours)
    "15m": 48,  // Predict next 48 intervals (12 hours)
    "1H": 24,   // Predict next 24 hours
    "4H": 12,   // Predict next 48 hours (12 × 4h)
    "1D": 5,    // Predict next 5 days
    "1W": 4,    // Predict next 4 weeks
    "1M": 3,    // Predict next 3 months
  };

  const { range, interval } = timeframeMap[timeframe];
  const horizon = horizonMap[timeframe];
  const { data: histData, isLoading: histLoading, error: histError } = useHistorical(selectedStock, range, interval);
  const { data: predData, isLoading: predLoading, error: predError } = usePrediction(selectedStock, horizon, range, interval);
  const { data: quoteData, isLoading: quoteLoading, error: quoteError } = useQuote(selectedStock);
  const { data: indicatorsData, isLoading: indicatorsLoading, error: indicatorsError } = useIndicators(selectedStock, range, interval);
  const { data: healthData, error: healthError } = useBackendHealth();

  const chartData = useMemo(() => {
    if (!histData || histData.length === 0) return [];
    // Limit visible bars for intraday intervals to keep chart readable
    const capByTimeframe: Partial<Record<Timeframe, number>> = {
      "5m": 100,
      "15m": 150,
      "1H": 200,
    };
    const cap = capByTimeframe[timeframe as keyof typeof capByTimeframe];
    const visible = cap ? histData.slice(-cap) : histData;

    // Return OHLCV for chart + predictions (predictions not sliced)
    const base = visible.map((d) => ({
      date: d.date,
      price: d.close,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));
    const forecast = (predData?.forecast || []).map((v, i) => ({
      date: `+${i + 1}`,
      price: v,
      predicted: true as const,
    }));
    return [...base, ...forecast];
  }, [histData, predData]);

  // Map API indicators to component format
  const indicators = useMemo(() => {
    if (!indicatorsData?.indicators) {
      // Return empty array or fallback while loading
      return [];
    }

    const ind = indicatorsData.indicators;
    const result: Array<{
      name: string;
      value: number | string;
      max?: number;
      icon: React.ReactNode;
      description: string;
    }> = [];

    // RSI
    if (ind.rsi) {
      const signalColor = 
        ind.rsi.signal === "overbought" ? "text-red-500" :
        ind.rsi.signal === "oversold" ? "text-green-500" : "text-primary";
      result.push({
        name: "RSI",
        value: ind.rsi.value,
        max: 100,
        icon: <Activity className={`h-5 w-5 ${signalColor}`} />,
        description: `Relative Strength Index (${ind.rsi.signal})`,
      });
    }

    // MACD
    if (ind.macd) {
      const trendColor = 
        ind.macd.trend === "bullish" ? "text-green-500" :
        ind.macd.trend === "bearish" ? "text-red-500" : "text-primary";
      const macdValue = ind.macd.histogram >= 0 ? `+${ind.macd.histogram.toFixed(4)}` : ind.macd.histogram.toFixed(4);
      result.push({
        name: "MACD",
        value: macdValue,
        icon: <TrendingUp className={`h-5 w-5 ${trendColor}`} />,
        description: `Moving Average Convergence (${ind.macd.trend})`,
      });
    }

    // Moving Averages
    if (ind.movingAverages) {
      const mas = ind.movingAverages;
      // Show available MAs (prioritize 50 and 200 for daily, or shorter for intraday)
      const maKeys = Object.keys(mas).sort((a, b) => {
        const aNum = parseInt(a.replace("sma", ""));
        const bNum = parseInt(b.replace("sma", ""));
        return aNum - bNum;
      });

      for (const key of maKeys) {
        const period = key.replace("sma", "");
        result.push({
          name: `MA ${period}-Period`,
          value: `$${mas[key].toFixed(2)}`,
          icon: <Move className="h-5 w-5 text-primary" />,
          description: `${period}-Period Simple Moving Average`,
        });
      }
    }

    // Bollinger Bands
    if (ind.bollingerBands) {
      const posLabel = 
        ind.bollingerBands.position === "upper" ? "Near Upper Band" :
        ind.bollingerBands.position === "lower" ? "Near Lower Band" : "Middle Band";
      result.push({
        name: "Bollinger Bands",
        value: `$${ind.bollingerBands.upper.toFixed(2)}`,
        icon: <Gauge className="h-5 w-5 text-primary" />,
        description: `Upper Band (${posLabel}, Width: ${ind.bollingerBands.width.toFixed(2)}%)`,
      });
    }

    // Volume
    if (ind.volume) {
      const volumeM = (ind.volume.current / 1_000_000).toFixed(1);
      const avgVolumeM = (ind.volume.average / 1_000_000).toFixed(1);
      result.push({
        name: "Volume",
        value: `${volumeM}M`,
        icon: <BarChart3 className="h-5 w-5 text-primary" />,
        description: `Trading Volume (Avg: ${avgVolumeM}M, Ratio: ${ind.volume.ratio.toFixed(2)}x)`,
      });
    }

    return result;
  }, [indicatorsData]);

  const mockMetrics = [
    {
      name: "Prediction Accuracy",
      value: "87.5%",
      icon: <Target className="h-6 w-6 text-primary" />,
      description: "Model accuracy over last 30 days",
    },
    {
      name: "Sharpe Ratio",
      value: "2.34",
      icon: <TrendingUp className="h-6 w-6 text-primary" />,
      description: "Risk-adjusted return measure",
    },
    {
      name: "Win Rate",
      value: "72%",
      icon: <Award className="h-6 w-6 text-primary" />,
      description: "Percentage of profitable trades",
    },
    {
      name: "Total Return",
      value: "+24.8%",
      icon: <DollarSign className="h-6 w-6 text-primary" />,
      description: "Cumulative return over time",
    },
  ];

  const mockWatchlist = [
    {
      symbol: "AAPL",
      name: "Apple Inc.",
      price: 162.45,
      change: 3.25,
      changePercent: 2.04,
      sparklineData: [158, 159, 157, 160, 162, 161, 162.45],
    },
    {
      symbol: "GOOGL",
      name: "Alphabet Inc.",
      price: 138.72,
      change: -1.28,
      changePercent: -0.91,
      sparklineData: [142, 140, 141, 139, 138, 139, 138.72],
    },
    {
      symbol: "MSFT",
      name: "Microsoft Corp.",
      price: 378.91,
      change: 5.67,
      changePercent: 1.52,
      sparklineData: [372, 375, 373, 376, 379, 377, 378.91],
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6 max-w-7xl space-y-6">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 space-y-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl font-bold mb-2">Stock Predictions Dashboard</h1>
                <p className="text-muted-foreground">
                  AI-powered market analysis and forecasts
                </p>
              </div>
              <TimeframeSelector
                selectedTimeframe={timeframe}
                onSelectTimeframe={setTimeframe}
              />
            </div>

            <HelpSection title="How It Works">
              <div className="space-y-3">
                <div>
                  <p className="font-medium mb-1">1. Select a Stock and Timeframe</p>
                  <p>Use the search bar above to find a stock by symbol (e.g., AAPL) or company name. Choose your preferred timeframe (5m, 15m, 1H, 1D, 1W, 1M) to view historical data and predictions.</p>
                </div>
                <div>
                  <p className="font-medium mb-1">2. Making Predictions</p>
                  <p>Predictions are automatically generated when you view a stock. The system uses an ensemble of AI models (LSTM, XGBoost, ARIMA, etc.) to forecast future prices. Each prediction is automatically saved to the database.</p>
                </div>
                <div>
                  <p className="font-medium mb-1">3. Understanding Results</p>
                  <p>The <strong>Prediction Summary</strong> shows the predicted price, confidence level, and trading signal (BUY/SELL/HOLD). The <strong>Price Chart</strong> displays historical data with predicted future prices in blue. <strong>Technical Indicators</strong> provide additional market insights.</p>
                </div>
                <div>
                  <p className="font-medium mb-1">4. Next Steps</p>
                  <p>After making predictions, go to the <strong>Prediction History</strong> page to evaluate how accurate your predictions were by comparing them with actual market prices.</p>
                </div>
              </div>
            </HelpSection>

            {/* Backend Connection Status */}
            {healthError && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertTitle>Backend Connection Failed</AlertTitle>
                <AlertDescription>
                  Cannot connect to the backend API. Please check:
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>Is the Railway backend service running?</li>
                    <li>Is VITE_API_BASE set correctly in Vercel environment variables?</li>
                    <li>Check Railway logs for backend errors</li>
                  </ul>
                  <div className="mt-2 text-xs">
                    Error: {healthError instanceof Error ? healthError.message : "Unknown error"}
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {healthData && !healthError && (
              <Alert className="border-green-500/50 bg-green-500/10">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <AlertTitle className="text-green-500">Backend Connected</AlertTitle>
                <AlertDescription className="text-green-500/80">
                  {healthData.service} v{healthData.version} is running
                </AlertDescription>
              </Alert>
            )}

            {/* API Error Messages */}
            {(quoteError || histError || predError) && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Data Loading Error</AlertTitle>
                <AlertDescription>
                  {quoteError && (
                    <div className="mb-2">
                      <strong>Quote Error:</strong> {quoteError instanceof Error ? quoteError.message : "Failed to load current price"}
                    </div>
                  )}
                  {histError && (
                    <div className="mb-2">
                      <strong>Historical Data Error:</strong> {histError instanceof Error ? histError.message : "Failed to load historical data"}
                    </div>
                  )}
                  {predError && (
                    <div>
                      <strong>Prediction Error:</strong> {predError instanceof Error ? predError.message : "Failed to generate prediction"}
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}

            <StockSelector
              selectedStock={selectedStock}
              onSelectStock={setSelectedStock}
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <PriceChart 
                  symbol={selectedStock} 
                  timeframe={timeframe} 
                  data={chartData}
                  liveQuote={quoteData ? { price: quoteData.price, previousClose: quoteData.previousClose } : undefined}
                />
              </div>
              <div>
                <PredictionSummary
                  currentPrice={quoteData?.price ?? histData?.at(-1)?.close ?? 0}
                  predictedPrice={predData?.predictedPrice ?? 0}
                  confidence={predData?.confidence ?? 0}
                  signal={
                    (predData?.predictedPrice ?? 0) > (quoteData?.price ?? histData?.at(-1)?.close ?? 0)
                      ? "BUY"
                      : (predData?.predictedPrice ?? 0) < (quoteData?.price ?? histData?.at(-1)?.close ?? 0)
                      ? "SELL"
                      : "HOLD"
                  }
                  models={predData?.models}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Technical Indicators</h2>
                {indicatorsLoading && (
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                )}
              </div>
              {indicators.length > 0 ? (
                <TechnicalIndicators indicators={indicators} />
              ) : indicatorsLoading ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                  <p>Loading indicators...</p>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <p>No indicators available</p>
                </div>
              )}
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
              <PerformanceMetrics metrics={mockMetrics} />
            </div>

            <RiskDisclaimer />
          </div>

          <div className="lg:w-80 space-y-6">
            <PredictionAccuracy symbol={selectedStock} interval={interval} />
            <Watchlist
              stocks={mockWatchlist}
              onRemove={(symbol) => console.log("Remove:", symbol)}
              onAdd={() => console.log("Add stock")}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
