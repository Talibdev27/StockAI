import { useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2 } from "lucide-react";
import StockSelector from "@/components/StockSelector";
import TimeRangeSelector, { TimeRange, DateRange } from "@/components/TimeRangeSelector";
import PredictionAccuracyChart from "@/components/PredictionAccuracyChart";
import PredictionMetricsCards from "@/components/PredictionMetricsCards";
import { usePredictionAccuracyData, DataSource, useDirectionalBias } from "@/hooks/useData";
import { filterByTimeRange, PredictionAccuracyPoint } from "@/lib/predictionAccuracyUtils";
import { usePredictionStats } from "@/hooks/useData";

export default function PredictionAccuracy() {
  const [symbol, setSymbol] = useState("AAPL");
  const [searchQuery, setSearchQuery] = useState("");
  const [dataSource, setDataSource] = useState<DataSource>("historical");
  const [interval, setInterval] = useState("1d");
  const [timeRange, setTimeRange] = useState<TimeRange>("30d");
  const [customDateRange, setCustomDateRange] = useState<DateRange>({
    from: undefined,
    to: undefined,
  });

  // Handle stock selection
  const handleSelectStock = (selectedSymbol: string) => {
    setSymbol(selectedSymbol);
    setSearchQuery("");
  };

  // Get prediction accuracy data
  const { data, isLoading, error } = usePredictionAccuracyData(
    dataSource,
    symbol,
    interval,
    dataSource === "backtest"
      ? {
          symbol,
          range: "1y",
          interval,
          strategy: "simple_signals",
        }
      : undefined
  );

  // Get stats for metrics cards
  const { data: stats } = usePredictionStats(symbol, interval);
  const { data: biasMetrics } = useDirectionalBias(symbol, interval, 90);

  // Filter data by time range
  const filteredPoints = useMemo(() => {
    if (!data?.points) return [];
    return filterByTimeRange(
      data.points,
      timeRange,
      customDateRange.from && customDateRange.to
        ? { from: customDateRange.from, to: customDateRange.to }
        : undefined
    );
  }, [data?.points, timeRange, customDateRange]);

  // Prepare stats for metrics cards
  const metricsStats = useMemo(() => {
    if (dataSource === "historical" && stats) {
      return stats;
    } else if (dataSource === "backtest" && data?.stats) {
      return data.stats;
    }
    return undefined;
  }, [dataSource, stats, data?.stats]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6 max-w-7xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Prediction Accuracy Analysis</h1>
          <p className="text-muted-foreground">
            Visualize prediction accuracy with actual vs predicted prices, confidence intervals, and detailed metrics.
          </p>
        </div>

        {/* Configuration Panel */}
        <Card className="p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Configuration</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label>Data Source</Label>
              <Tabs value={dataSource} onValueChange={(v) => setDataSource(v as DataSource)}>
                <TabsList className="w-full">
                  <TabsTrigger value="historical" className="flex-1">
                    Historical
                  </TabsTrigger>
                  <TabsTrigger value="backtest" className="flex-1">
                    Backtest
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div>
              <Label>Stock Symbol</Label>
              <StockSelector
                selectedStock={symbol}
                onSelectStock={handleSelectStock}
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
              />
            </div>

            <div>
              <Label>Interval</Label>
              <Select value={interval} onValueChange={setInterval}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1d">Daily</SelectItem>
                  <SelectItem value="1wk">Weekly</SelectItem>
                  <SelectItem value="1mo">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Time Range</Label>
              <TimeRangeSelector
                selectedRange={timeRange}
                onRangeChange={setTimeRange}
                customDateRange={customDateRange}
                onCustomDateRangeChange={setCustomDateRange}
              />
            </div>
          </div>
        </Card>

        {/* Error Display */}
        {error && (
          <Card className="p-4 bg-destructive/10 border-destructive">
            <p className="text-destructive">
              {error instanceof Error ? error.message : "Failed to load prediction accuracy data. Please try again."}
            </p>
          </Card>
        )}

        {/* Loading State */}
        {isLoading && (
          <Card className="p-6">
            <div className="flex items-center justify-center space-x-2">
              <Loader2 className="h-6 w-6 animate-spin" />
              <span>Loading prediction accuracy data...</span>
            </div>
          </Card>
        )}

        {/* Results Display */}
        {!isLoading && !error && filteredPoints.length > 0 && (
          <div className="space-y-6">
            {/* Metrics Cards */}
            <PredictionMetricsCards stats={metricsStats} isLoading={isLoading} />

            {/* Directional Bias Summary */}
            {biasMetrics && (
              <Card className="p-4">
                <h3 className="text-lg font-semibold mb-2">Directional Bias (lookback ~90 days)</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="font-medium mb-1">Predicted Direction Mix</p>
                    <p className="text-muted-foreground">
                      Up: {biasMetrics.direction_counts.up ?? 0} · Down: {biasMetrics.direction_counts.down ?? 0} ·
                      Neutral: {biasMetrics.direction_counts.neutral ?? 0}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Down-Bias Error (when predicting Down)</p>
                    <p className="text-muted-foreground">
                      Avg error: {biasMetrics.down_bias.avg_error_percent.toFixed(2)}% ·
                      Avg realized move: {biasMetrics.down_bias.avg_realized_move_percent.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="font-medium mb-1">High-Confidence Down Calibration</p>
                    <p className="text-muted-foreground">
                      80–90% bucket hit-rate:{" "}
                      {biasMetrics.confidence_buckets["80-90"]
                        ? `${biasMetrics.confidence_buckets["80-90"].down_hit_rate.toFixed(1)}%`
                        : "n/a"}
                      ; 90%+ bucket hit-rate:{" "}
                      {biasMetrics.confidence_buckets["90+"]
                        ? `${biasMetrics.confidence_buckets["90+"].down_hit_rate.toFixed(1)}%`
                        : "n/a"}
                    </p>
                  </div>
                </div>
              </Card>
            )}

            {/* Chart */}
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Prediction Accuracy Chart</h2>
              </div>
              <div className="mb-4 p-3 bg-muted/50 rounded-lg border border-border/50">
                <div className="flex items-center gap-6 flex-wrap text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-1 bg-blue-400 rounded" />
                    <span className="font-medium">Actual Price</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-1 bg-orange-400 border-dashed border-2 border-orange-400" />
                    <span className="font-medium">Predicted Price</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-3 bg-slate-400/30 rounded-sm" />
                    <span className="font-medium">Confidence Interval</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500 shadow-sm" />
                    <span>Accurate (≤2%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500 shadow-sm" />
                    <span>Moderate (2-5%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500 shadow-sm" />
                    <span>Inaccurate (&gt;5%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-sm" />
                    <span>Pending</span>
                  </div>
                </div>
              </div>
              <div className="rounded-lg overflow-hidden border border-border/30">
                <PredictionAccuracyChart data={filteredPoints} symbol={symbol} />
              </div>
            </Card>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && (!data || filteredPoints.length === 0) && (
          <Card className="p-6">
            <div className="text-center text-muted-foreground">
              <p>No prediction accuracy data available.</p>
              <p className="text-sm mt-2">
                {dataSource === "historical"
                  ? "Try selecting a different stock or ensure predictions have been made and evaluated."
                  : "Try running a backtest or selecting a different stock."}
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

