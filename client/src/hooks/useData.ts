import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";

export function useBackendHealth() {
  return useQuery({
    queryKey: ["backend-health"],
    queryFn: () => apiGet<{ status: string; service: string; version: string }>("/api/health"),
    retry: 1,
    retryDelay: 2000,
    refetchInterval: 30000, // Check every 30 seconds
    staleTime: 1000 * 10, // 10 seconds
  });
}

export function useStocks() {
  return useQuery({
    queryKey: ["stocks"],
    queryFn: () => apiGet<Array<{ symbol: string; name: string }>>("/api/stocks"),
    staleTime: 1000 * 60 * 5,
    retry: 2, // Retry up to 2 times on failure
    retryDelay: 1000, // Wait 1 second between retries
    refetchOnWindowFocus: false, // Don't refetch on window focus
  });
}

export function useHistorical(symbol: string, range = "1y", interval = "1d") {
  return useQuery({
    queryKey: ["historical", symbol, range, interval],
    queryFn: () =>
      apiGet<Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }>>(
        `/api/historical/${symbol}?range=${range}&interval=${interval}`,
      ),
    enabled: !!symbol,
    staleTime: 1000 * 60 * 10,
  });
}

export function usePrediction(symbol: string, horizon = 5, range = "1y", interval = "1d") {
  return useQuery({
    queryKey: ["prediction", symbol, horizon, range, interval],
    queryFn: () =>
      apiGet<{
        symbol: string;
        predictedPrice: number;
        confidence: number;
        forecast: number[];
        models?: Record<string, { prediction: number; confidence: number; weight: number; metrics?: any; status?: string }>;
        warnings?: string[];
      }>(
      `/api/predict/${symbol}?horizon=${horizon}&range=${range}&interval=${interval}`,
      ),
    enabled: !!symbol,
    staleTime: 1000 * 60 * 5,
  });
}

export function useQuote(symbol: string) {
  return useQuery({
    queryKey: ["quote", symbol],
    queryFn: () => apiGet<{ symbol: string; price: number; previousClose: number; currency: string; change: number; changePercent: number }>(
      `/api/quote/${symbol}`,
    ),
    enabled: !!symbol,
    refetchInterval: 30000, // refresh every 30s
    staleTime: 1000 * 30, // 30 seconds
  });
}

export interface BacktestParams {
  symbol: string;
  range?: string;
  interval?: string;
  strategy?: string;
  initialCapital?: number;
  commission?: number;
  positionSize?: number;
  threshold?: number;
  slippage?: number;
  slippageType?: string;
}

export interface BacktestResult {
  symbol: string;
  period: { start: string; end: string };
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  metrics: {
    pnl: number;
    returnPercent: number;
    sharpeRatio: number;
    winRate: number;
    maxDrawdown: number;
    totalTrades: number;
    avgWin: number;
    avgLoss: number;
    totalSlippage?: number;
    slippageImpactPercent?: number;
  };
  equityCurve: Array<{ date: string; value: number; price: number }>;
  trades: Array<{ 
    date: string; 
    type: "buy" | "sell"; 
    price: number; 
    execution_price?: number;
    shares: number; 
    pnl?: number; 
    cost?: number;
    slippage_cost?: number;
    slippage_pct?: number;
  }>;
}

export function useBacktest(params: BacktestParams, enabled = true) {
  const {
    symbol,
    range = "1y",
    interval = "1d",
    strategy = "simple_signals",
    initialCapital = 10000,
    commission = 0.001,
    positionSize = 1.0,
    threshold = 0.0,
    slippage = 0.0005,
    slippageType = "hybrid",
  } = params;

  return useQuery({
    queryKey: ["backtest", symbol, range, interval, strategy, initialCapital, commission, positionSize, threshold, slippage, slippageType],
    queryFn: () => {
      const queryParams = new URLSearchParams({
        range,
        interval,
        strategy,
        initial_capital: initialCapital.toString(),
        commission: commission.toString(),
        position_size: positionSize.toString(),
        threshold: threshold.toString(),
        slippage: slippage.toString(),
        slippage_type: slippageType,
      });
      return apiGet<BacktestResult>(`/api/backtest/${symbol}?${queryParams}`);
    },
    enabled: enabled && !!symbol,
    staleTime: 1000 * 60 * 10, // 10 minutes
    retry: 1, // Backtests can be slow, only retry once
  });
}

export interface IndicatorsData {
  symbol: string;
  lastUpdated: string;
  indicators: {
    rsi?: {
      value: number;
      signal: "overbought" | "oversold" | "neutral";
      period: number;
    };
    macd?: {
      macd: number;
      signal: number;
      histogram: number;
      trend: "bullish" | "bearish" | "neutral";
    };
    movingAverages?: {
      [key: string]: number; // e.g., sma20, sma50, sma200
    };
    bollingerBands?: {
      upper: number;
      middle: number;
      lower: number;
      width: number;
      position: "upper" | "middle" | "lower";
    };
    volume?: {
      current: number;
      average: number;
      ratio: number;
    };
  };
}

export function useIndicators(symbol: string, range = "1y", interval = "1d") {
  return useQuery({
    queryKey: ["indicators", symbol, range, interval],
    queryFn: () => apiGet<IndicatorsData>(`/api/indicators/${symbol}?range=${range}&interval=${interval}`),
    enabled: !!symbol,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export interface PredictionHistoryItem {
  id: number;
  symbol: string;
  timestamp: string;
  interval: string;
  horizon: number;
  current_price: number;
  predicted_price: number;
  confidence: number;
  evaluated: boolean;
  model_breakdown?: Record<string, any>;
  actual_price?: number;
  error?: number;
  error_percent?: number;
  direction_actual?: "up" | "down" | "neutral";
  direction_predicted?: "up" | "down" | "neutral";
  correct?: boolean;
  evaluated_at?: string;
}

export interface PredictionStats {
  total: number;
  direction_accuracy: number;
  avg_error: number;
  avg_error_percent: number;
  rmse: number;
  mae: number;
}

export function usePredictionHistory(symbol: string | undefined, limit = 100, offset = 0, interval?: string) {
  return useQuery({
    queryKey: ["prediction-history", symbol || "all", limit, offset, interval],
    queryFn: () => {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
      });
      if (interval) params.append("interval", interval);
      const url = symbol 
        ? `/api/predictions/history/${symbol}?${params}`
        : `/api/predictions/history?${params}`;
      return apiGet<PredictionHistoryItem[]>(url);
    },
    enabled: true,
    staleTime: 1000 * 60 * 2, // 2 minutes
  });
}

export function usePredictionStats(symbol: string | undefined, interval?: string) {
  return useQuery({
    queryKey: ["prediction-stats", symbol || "all", interval],
    queryFn: () => {
      const params = interval ? new URLSearchParams({ interval }) : "";
      const url = symbol 
        ? `/api/predictions/stats/${symbol}${params ? `?${params}` : ""}`
        : `/api/predictions/stats${params ? `?${params}` : ""}`;
      return apiGet<PredictionStats>(url);
    },
    enabled: true,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export interface EvaluationResult {
  evaluated: number;
  errors: number;
  total: number;
  message?: string;
}

export function useEvaluation(symbol?: string, maxPredictions = 100) {
  return useQuery({
    queryKey: ["evaluate-predictions", symbol, maxPredictions],
    queryFn: () => {
      const params = new URLSearchParams({
        max: maxPredictions.toString(),
      });
      if (symbol) params.append("symbol", symbol);
      return apiGet<EvaluationResult>(`/api/predictions/evaluate?${params}`);
    },
    enabled: false, // Manual trigger only
    staleTime: 0,
    retry: false,
  });
}

export interface TradingPerformanceMetrics {
  prediction_accuracy: number;
  sharpe_ratio: number;
  win_rate: number;
  total_return: number;
  total_predictions: number;
  profitable_predictions: number;
}

export function useTradingPerformance(symbol?: string, interval?: string, lookbackDays = 30) {
  return useQuery({
    queryKey: ["trading-performance", symbol || "all", interval, lookbackDays],
    queryFn: () => {
      const params = new URLSearchParams({
        lookback_days: lookbackDays.toString(),
      });
      if (interval) params.append("interval", interval);
      const url = symbol 
        ? `/api/predictions/performance/${symbol}?${params}`
        : `/api/predictions/performance?${params}`;
      return apiGet<TradingPerformanceMetrics>(url);
    },
    enabled: true,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}


