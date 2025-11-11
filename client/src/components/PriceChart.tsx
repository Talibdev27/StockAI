import { Card } from "@/components/ui/card";
import { Timeframe } from "./TimeframeSelector";
import { useEffect, useMemo, useRef, useState } from "react";
import { createChart, type ISeriesApi, type IChartApi, LineStyle } from "lightweight-charts";
import { detectPatterns, type Candle, type PatternName, type DetectedPattern } from "@/lib/candlestickPatterns";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Maximize2, Minimize2 } from "lucide-react";
import PatternControls from "./PatternControls";
import RecentSignalsPanel from "./RecentSignalsPanel";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface PriceChartProps {
  symbol: string;
  timeframe: Timeframe;
  data: Array<{
    date: string;
    price: number;
    predicted?: boolean;
    open?: number;
    high?: number;
    low?: number;
    close?: number;
    volume?: number;
  }>;
  liveQuote?: {
    price: number;
    previousClose: number;
  };
}

type ChartMode = "line" | "candles";

export default function PriceChart({ symbol, timeframe, data, liveQuote }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<"Candlestick"> | ISeriesApi<"Line"> | null>(null);
  const predictSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const currentPriceLineRef = useRef<any>(null);
  const prevCloseLineRef = useRef<any>(null);

  const [mode, setMode] = useState<ChartMode>("candles");
  const [showPatterns, setShowPatterns] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [enabledPatterns, setEnabledPatterns] = useState<Set<PatternName>>(
    new Set<PatternName>(["Doji", "Hammer", "Shooting Star", "Bullish Engulfing", "Bearish Engulfing"])
  );
  const [strengthThreshold, setStrengthThreshold] = useState(60);
  const [compactMode, setCompactMode] = useState(false);
  const [pinnedSignals, setPinnedSignals] = useState<Set<number>>(new Set());
  const [hiddenPatternTypes, setHiddenPatternTypes] = useState<Set<PatternName>>(new Set());

  const ohlc: Array<Candle & { predicted?: boolean }> = useMemo(() => {
    return data.map((d) => ({
      date: d.date,
      open: d.open ?? d.price,
      high: d.high ?? d.price,
      low: d.low ?? d.price,
      close: d.close ?? d.price,
      volume: d.volume ?? 0,
      ...(d.predicted ? { predicted: true } : {}),
    }));
  }, [data]);

  const predictions = useMemo(
    () => data.filter((d) => d.predicted).map((d) => ({ time: d.date as any, value: d.price })),
    [data],
  );

  // Helpers
  const isValidDate = (s: string) => /^\d{4}-\d{2}-\d{2}/.test(s); // matches YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
  const addDays = (date: string, n: number) => {
    const d = new Date(date);
    d.setDate(d.getDate() + n);
    return d.toISOString().slice(0, 10);
  };
  const formatTime = (s: string) => {
    // lightweight-charts needs YYYY-MM-DD for daily or unix timestamp for intraday
    // For simplicity, pass date strings as-is and let the library handle
    if (s.includes(" ")) return s.split(" ")[0]; // strip time for now
    return s;
  };

  // Only real historical OHLC with valid dates
  const hist = useMemo(
    () => ohlc.filter((c) => !c.predicted && isValidDate(c.date)),
    [ohlc],
  );

  // Proper-dated prediction line: next days after last historical
  const lastHistDate = hist.length ? hist[hist.length - 1].date : undefined;
  const predLine = useMemo(() => {
    if (!lastHistDate) return [] as Array<{ time: string; value: number }>;
    const base = data.filter((d) => d.predicted).map((d) => d.price);
    return base.map((v, i) => ({ time: addDays(lastHistDate, i + 1), value: v }));
  }, [lastHistDate, data]);

  // Detect patterns only from real historical candles
  const allPatterns = useMemo(() => (showPatterns ? detectPatterns(hist) : []), [hist, showPatterns]);
  
  // Filter patterns based on settings
  const filteredPatterns = useMemo(() => {
    return allPatterns.filter(
      (p) =>
        enabledPatterns.has(p.name) &&
        !hiddenPatternTypes.has(p.name) &&
        p.strength >= strengthThreshold
    );
  }, [allPatterns, enabledPatterns, hiddenPatternTypes, strengthThreshold]);
  
  // Get recent strong signals for panel (last 10, sorted by strength)
  const recentSignals = useMemo(() => {
    return filteredPatterns
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 10);
  }, [filteredPatterns]);

  // Fullscreen functionality
  const enterFullscreen = async () => {
    if (containerRef.current && !document.fullscreenElement) {
      try {
        await containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } catch (err) {
        console.error("Error entering fullscreen:", err);
      }
    }
  };

  const exitFullscreen = async () => {
    if (document.fullscreenElement) {
      try {
        await document.exitFullscreen();
        setIsFullscreen(false);
      } catch (err) {
        console.error("Error exiting fullscreen:", err);
      }
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Price line helpers
  const addPriceLines = (series: any) => {
    if (!series || hist.length < 2) return;

    // Remove existing price lines
    if (currentPriceLineRef.current) {
      series.removePriceLine(currentPriceLineRef.current);
    }
    if (prevCloseLineRef.current) {
      series.removePriceLine(prevCloseLineRef.current);
    }

    // Use live quote data if available, otherwise fall back to historical data
    const currentPrice = liveQuote?.price ?? hist[hist.length - 1]?.close;
    const prevClose = liveQuote?.previousClose ?? hist[hist.length - 2]?.close;
    const priceChange = currentPrice && prevClose ? ((currentPrice - prevClose) / prevClose) * 100 : 0;

    // Current price line
    if (currentPrice) {
      currentPriceLineRef.current = series.createPriceLine({
        price: currentPrice,
        color: priceChange >= 0 ? "#26a69a" : "#ef5350",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Current: $${currentPrice.toFixed(2)}`,
      });
    }

    // Previous close line
    if (prevClose) {
      prevCloseLineRef.current = series.createPriceLine({
        price: prevClose,
        color: "#6B7280",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: `Prev Close: $${prevClose.toFixed(2)}`,
      });
    }
  };

  useEffect(() => {
    if (!containerRef.current) return;
    if (chartRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: { background: { color: "#131722" }, textColor: "#9BA3AF" },
      grid: { vertLines: { color: "#2A2E39" }, horzLines: { color: "#2A2E39" } },
      rightPriceScale: { borderColor: "#2A2E39" },
      timeScale: { borderColor: "#2A2E39" },
      autoSize: true,
      height: isFullscreen ? window.innerHeight * 0.95 : 380,
      handleScroll: true,
      handleScale: true,
    } as any);

    chartRef.current = chart;
    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    if (mainSeriesRef.current) {
      chart.removeSeries(mainSeriesRef.current);
      mainSeriesRef.current = null as any;
    }
    if (predictSeriesRef.current) {
      chart.removeSeries(predictSeriesRef.current);
      predictSeriesRef.current = null;
    }

    try {
      if (mode === "candles") {
        const series = (chart as any).addCandlestickSeries({
          upColor: "#26a69a",
          downColor: "#ef5350",
          borderUpColor: "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor: "#26a69a",
          wickDownColor: "#ef5350",
        });
        mainSeriesRef.current = series;
      } else {
        const series = (chart as any).addLineSeries({ color: "#60A5FA", lineWidth: 2 });
        mainSeriesRef.current = series;
      }

      predictSeriesRef.current = (chart as any).addLineSeries({ color: "#3B82F6", lineWidth: 2 });
    } catch (error) {
      console.error("Error adding series:", error);
    }
  }, [mode]);

  // Helper to convert datetime string to timestamp
  const dateToTimestamp = (dateStr: string): number => {
    return Math.floor(new Date(dateStr).getTime() / 1000);
  };

  useEffect(() => {
    const chart = chartRef.current;
    const main = mainSeriesRef.current;
    const pred = predictSeriesRef.current;
    if (!chart || !main || !ohlc.length) return;

    // Detect if we have intraday data (contains time)
    const hasTime = hist.length > 0 && hist[0].date.includes(':');

    if (mode === "candles") {
      if (hasTime) {
        // For intraday: convert to timestamps
        (main as ISeriesApi<"Candlestick">).setData(
          hist.map((c) => ({ 
            time: dateToTimestamp(c.date) as any, 
            open: c.open, 
            high: c.high, 
            low: c.low, 
            close: c.close 
          })),
        );
      } else {
        // For daily+: use date strings as-is
        (main as ISeriesApi<"Candlestick">).setData(
          hist.map((c) => ({ time: c.date as any, open: c.open, high: c.high, low: c.low, close: c.close })),
        );
      }
    } else {
      if (hasTime) {
        // For intraday: convert to timestamps
        (main as ISeriesApi<"Line">).setData(hist.map((c) => ({ time: dateToTimestamp(c.date) as any, value: c.close })));
      } else {
        // For daily+: use date strings as-is
        (main as ISeriesApi<"Line">).setData(hist.map((c) => ({ time: c.date as any, value: c.close })));
      }
    }

    pred?.setData(predLine as any);
    (chart as any).timeScale().fitContent();

    // Add price lines
    addPriceLines(main);

    if (showPatterns && mode === "candles" && !compactMode) {
      // Sort filtered patterns by index (time order) before creating markers
      const sortedPatterns = [...filteredPatterns].sort((a, b) => a.index - b.index);
      
      const regularMarkers = sortedPatterns
        .filter((p) => !pinnedSignals.has(p.index))
        .slice(-100) // Limit to last 100 for performance
        .map((p) => ({
          time: hasTime ? dateToTimestamp(hist[p.index].date) as any : hist[p.index].date as any,
          position: p.direction === "bullish" ? "belowBar" : p.direction === "bearish" ? "aboveBar" : "inBar",
          color: p.direction === "bullish" ? "#26a69a" : p.direction === "bearish" ? "#ef5350" : "#9BA3AF",
          shape: "circle", // Use circle icon for compact display
          text: `${p.name} (${p.strength})`, // Show name and strength in tooltip
          size: 1, // Small size
        }));
      
      // Add pinned signals with gold color
      const pinned = sortedPatterns
        .filter((p) => pinnedSignals.has(p.index))
        .map((p) => ({
          time: hasTime ? dateToTimestamp(hist[p.index].date) as any : hist[p.index].date as any,
          position: p.direction === "bullish" ? "belowBar" : p.direction === "bearish" ? "aboveBar" : "inBar",
          color: "#FFD700", // Gold for pinned
          shape: "arrowUp",
          text: `📌 ${p.name} (${p.strength})`,
          size: 2,
        }));
      
      // Combine and sort by time (required by lightweight-charts)
      const allMarkers = [...regularMarkers, ...pinned];
      allMarkers.sort((a, b) => {
        const timeA = typeof a.time === "number" ? a.time : new Date(a.time as string).getTime() / 1000;
        const timeB = typeof b.time === "number" ? b.time : new Date(b.time as string).getTime() / 1000;
        return timeA - timeB;
      });
      
      (main as any).setMarkers(allMarkers);
    } else if (mode === "candles") {
      (main as any).setMarkers([]);
    }
  }, [ohlc, predictions, mode, showPatterns, filteredPatterns, hist, predLine, compactMode, pinnedSignals]);

  // Handlers for pattern controls
  const handleTogglePattern = (name: PatternName) => {
    const newSet = new Set(enabledPatterns);
    if (newSet.has(name)) {
      newSet.delete(name);
    } else {
      newSet.add(name);
    }
    setEnabledPatterns(newSet);
  };

  const handlePin = (index: number) => {
    const newSet = new Set(pinnedSignals);
    newSet.add(index);
    setPinnedSignals(newSet);
  };

  const handleUnpin = (index: number) => {
    const newSet = new Set(pinnedSignals);
    newSet.delete(index);
    setPinnedSignals(newSet);
  };

  const handleHideType: (name: PatternName) => void = (name) => {
    const newSet = new Set(hiddenPatternTypes);
    newSet.add(name);
    setHiddenPatternTypes(newSet);
  };

  const handleCreateAlert = (signal: DetectedPattern) => {
    // TODO: Implement alert creation
    alert(`Alert created for ${signal.name} (Strength: ${signal.strength})`);
  };

  return (
    <TooltipProvider>
    <Card className={`p-6 ${isFullscreen ? "fixed inset-0 z-50 bg-background" : ""}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">Price Chart</h3>
          <p className="text-sm text-muted-foreground">
            {symbol} - {timeframe} Timeframe
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <span>Line</span>
            <Switch checked={mode === "candles"} onCheckedChange={(v) => setMode(v ? "candles" : "line")} />
            <span>Candles</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span>Patterns</span>
            <Switch checked={showPatterns} onCheckedChange={setShowPatterns} />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={isFullscreen ? exitFullscreen : enterFullscreen}
            className="flex items-center gap-2"
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            {isFullscreen ? "Exit Fullscreen" : "Full Chart"}
          </Button>
        </div>
        </div>

      <div 
        ref={containerRef} 
        className="w-full" 
        style={{ height: isFullscreen ? `${window.innerHeight * 0.95}px` : "380px" }} 
      />

      <div className="flex items-center justify-center gap-4 text-sm mt-3">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#26a69a]" />
          <span className="text-muted-foreground">Bullish</span>
        </div>
          <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#ef5350]" />
          <span className="text-muted-foreground">Bearish</span>
          </div>
          <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#60A5FA]" />
            <span className="text-muted-foreground">Predicted</span>
        </div>
      </div>

      {showPatterns && (
        <div className="mt-4 space-y-4">
          <PatternControls
            enabledPatterns={enabledPatterns}
            onTogglePattern={handleTogglePattern}
            strengthThreshold={strengthThreshold}
            onStrengthThresholdChange={setStrengthThreshold}
            compactMode={compactMode}
            onCompactModeChange={setCompactMode}
          />
          <RecentSignalsPanel
            signals={recentSignals}
            pinnedSignals={pinnedSignals}
            onPin={handlePin}
            onUnpin={handleUnpin}
            onHideType={handleHideType}
            onCreateAlert={handleCreateAlert}
          />
        </div>
      )}
    </Card>
    </TooltipProvider>
  );
}
