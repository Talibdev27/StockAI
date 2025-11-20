import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import StockSelector from "@/components/StockSelector";
import { useBacktest } from "@/hooks/useData";
import { Loader2, TrendingUp, TrendingDown, DollarSign, Target, BarChart3, Activity } from "lucide-react";
import { createChart, ColorType, IChartApi } from "lightweight-charts";
import { useRef, useEffect } from "react";

export default function Backtest() {
  const [symbol, setSymbol] = useState("AAPL");
  const [searchQuery, setSearchQuery] = useState("");
  const [range, setRange] = useState("1y");
  const [interval, setInterval] = useState("1d");
  const [strategy, setStrategy] = useState("simple_signals");
  const [initialCapital, setInitialCapital] = useState(10000);
  const [commission, setCommission] = useState(0.001);
  const [positionSize, setPositionSize] = useState(1.0);
  const [threshold, setThreshold] = useState(0.0);
  const [slippage, setSlippage] = useState(0.0005);
  const [slippageType, setSlippageType] = useState("hybrid");
  const [runBacktest, setRunBacktest] = useState(false);

  const { data: results, isLoading, error } = useBacktest(
    {
      symbol,
      range,
      interval,
      strategy,
      initialCapital,
      commission,
      positionSize,
      threshold,
      slippage,
      slippageType,
    },
    runBacktest
  );

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Handle stock selection
  const handleSelectStock = (selectedSymbol: string) => {
    setSymbol(selectedSymbol);
    setSearchQuery("");
    setRunBacktest(false);
  };

  // Run backtest
  const handleRunBacktest = () => {
    setRunBacktest(true);
  };

  // Render equity curve chart
  useEffect(() => {
    if (!results || !chartContainerRef.current) return;

    // Clean up existing chart
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "rgb(226, 232, 240)",
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.1)" },
        horzLines: { color: "rgba(148, 163, 184, 0.1)" },
      },
    });

    // Add equity curve series
    const equitySeries = chart.addLineSeries({
      color: "#3b82f6",
      lineWidth: 2,
      title: "Portfolio Value",
    });

    // Add price series
    const priceSeries = chart.addLineSeries({
      color: "#64748b",
      lineWidth: 1,
      title: "Stock Price",
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });

    // Format data
    const equityData = results.equityCurve.map((point) => ({
      time: point.date,
      value: point.value,
    }));

    const priceData = results.equityCurve.map((point) => ({
      time: point.date,
      value: point.price,
    }));

    equitySeries.setData(equityData);
    priceSeries.setData(priceData);

    // Add buy/sell markers
    const markers = results.trades.map((trade, idx) => ({
      time: trade.date,
      position: trade.type === "buy" ? "belowBar" : "aboveBar",
      color: trade.type === "buy" ? "#10b981" : "#ef4444",
      shape: trade.type === "buy" ? "arrowUp" : "arrowDown",
      text: trade.type === "buy" ? "Buy" : "Sell",
    }));

    equitySeries.setMarkers(markers as any);

    chart.timeScale().fitContent();
    chartRef.current = chart;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [results]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6 max-w-7xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Backtest Results</h1>
          <p className="text-muted-foreground">
            Run historical simulations to evaluate trading strategies based on ML predictions.
          </p>
        </div>

        {/* Configuration Panel */}
        <Card className="p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Backtest Configuration</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <Label>Date Range</Label>
              <Select value={range} onValueChange={setRange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1y">1 Year</SelectItem>
                  <SelectItem value="2y">2 Years</SelectItem>
                  <SelectItem value="3y">3 Years</SelectItem>
                  <SelectItem value="5y">5 Years</SelectItem>
                </SelectContent>
              </Select>
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
              <Label>Strategy</Label>
              <Select value={strategy} onValueChange={setStrategy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="simple_signals">Simple Signals (ML)</SelectItem>
                  <SelectItem value="threshold">Threshold (ML)</SelectItem>
                  <SelectItem value="momentum">Momentum (ML)</SelectItem>
                  <SelectItem value="buy_and_hold">Buy-and-Hold (Benchmark)</SelectItem>
                  <SelectItem value="random">Random Trading (Benchmark)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Initial Capital ($)</Label>
              <Input
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(Number(e.target.value))}
                min={1000}
                step={1000}
              />
            </div>

            <div>
              <Label>Commission (0.001 = 0.1%)</Label>
              <Input
                type="number"
                value={commission}
                onChange={(e) => setCommission(Number(e.target.value))}
                min={0}
                max={0.01}
                step={0.0001}
              />
            </div>

            <div>
              <Label>Position Size (1.0 = 100%)</Label>
              <Input
                type="number"
                value={positionSize}
                onChange={(e) => setPositionSize(Number(e.target.value))}
                min={0.1}
                max={1.0}
                step={0.1}
              />
            </div>

            <div>
              <Label>Threshold (for threshold/momentum strategies)</Label>
              <Input
                type="number"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                min={0}
                max={0.1}
                step={0.001}
              />
            </div>

            <div>
              <Label>Slippage (0.0005 = 0.05%)</Label>
              <Input
                type="number"
                value={slippage}
                onChange={(e) => setSlippage(Number(e.target.value))}
                min={0}
                max={0.01}
                step={0.0001}
              />
            </div>

            <div>
              <Label>Slippage Type</Label>
              <Select value={slippageType} onValueChange={setSlippageType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">Fixed</SelectItem>
                  <SelectItem value="volatility">Volatility-Based</SelectItem>
                  <SelectItem value="hybrid">Hybrid (Recommended)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button
            onClick={handleRunBacktest}
            disabled={isLoading || !symbol}
            className="w-full md:w-auto"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Backtest...
              </>
            ) : (
              "Run Backtest"
            )}
          </Button>
        </Card>

        {/* Error Display */}
        {error && (
          <Card className="p-4 bg-destructive/10 border-destructive">
            <p className="text-destructive">
              {error instanceof Error ? error.message : "Failed to run backtest. Please try again."}
            </p>
          </Card>
        )}

        {/* Results Display */}
        {results && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total P&L</p>
                    <p className={`text-2xl font-bold ${results.metrics.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                      ${results.metrics.pnl.toLocaleString()}
                    </p>
                  </div>
                  <DollarSign className="h-8 w-8 text-muted-foreground" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Return %</p>
                    <p className={`text-2xl font-bold ${results.metrics.returnPercent >= 0 ? "text-green-500" : "text-red-500"}`}>
                      {results.metrics.returnPercent.toFixed(2)}%
                    </p>
                  </div>
                  {results.metrics.returnPercent >= 0 ? (
                    <TrendingUp className="h-8 w-8 text-green-500" />
                  ) : (
                    <TrendingDown className="h-8 w-8 text-red-500" />
                  )}
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Sharpe Ratio</p>
                    <p className="text-2xl font-bold">{results.metrics.sharpeRatio.toFixed(2)}</p>
                  </div>
                  <Target className="h-8 w-8 text-muted-foreground" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Win Rate</p>
                    <p className="text-2xl font-bold">{(results.metrics.winRate * 100).toFixed(1)}%</p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-muted-foreground" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Max Drawdown</p>
                    <p className="text-2xl font-bold text-red-500">
                      {(results.metrics.maxDrawdown * 100).toFixed(2)}%
                    </p>
                  </div>
                  <TrendingDown className="h-8 w-8 text-red-500" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Trades</p>
                    <p className="text-2xl font-bold">{results.metrics.totalTrades}</p>
                  </div>
                  <Activity className="h-8 w-8 text-muted-foreground" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Win</p>
                    <p className="text-2xl font-bold text-green-500">
                      ${results.metrics.avgWin.toFixed(2)}
                    </p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-green-500" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Loss</p>
                    <p className="text-2xl font-bold text-red-500">
                      ${results.metrics.avgLoss.toFixed(2)}
                    </p>
                  </div>
                  <TrendingDown className="h-8 w-8 text-red-500" />
                </div>
              </Card>

              {results.metrics.totalSlippage !== undefined && (
                <Card className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Total Slippage</p>
                      <p className="text-2xl font-bold text-orange-500">
                        ${results.metrics.totalSlippage.toFixed(2)}
                      </p>
                      {results.metrics.slippageImpactPercent !== undefined && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {results.metrics.slippageImpactPercent.toFixed(3)}% of capital
                        </p>
                      )}
                    </div>
                    <Activity className="h-8 w-8 text-orange-500" />
                  </div>
                </Card>
              )}
            </div>

            {/* Equity Curve Chart */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Equity Curve</h2>
              <div ref={chartContainerRef} className="w-full" style={{ height: "400px" }} />
            </Card>

            {/* Trades Table */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Trades</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Date</th>
                      <th className="text-left p-2">Type</th>
                      <th className="text-right p-2">Price</th>
                      <th className="text-right p-2">Exec. Price</th>
                      <th className="text-right p-2">Shares</th>
                      <th className="text-right p-2">Slippage</th>
                      <th className="text-right p-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.trades.map((trade, idx) => (
                      <tr key={idx} className="border-b">
                        <td className="p-2">{trade.date}</td>
                        <td className="p-2">
                          <span
                            className={`px-2 py-1 rounded ${
                              trade.type === "buy"
                                ? "bg-green-500/20 text-green-500"
                                : "bg-red-500/20 text-red-500"
                            }`}
                          >
                            {trade.type.toUpperCase()}
                          </span>
                        </td>
                        <td className="text-right p-2">${trade.price.toFixed(2)}</td>
                        <td className="text-right p-2">
                          {trade.execution_price !== undefined ? (
                            <>
                              ${trade.execution_price.toFixed(2)}
                              {trade.slippage_pct !== undefined && (
                                <span className={`text-xs ml-1 ${trade.type === "buy" ? "text-red-400" : "text-red-400"}`}>
                                  ({trade.slippage_pct.toFixed(3)}%)
                                </span>
                              )}
                            </>
                          ) : (
                            "-"
                          )}
                        </td>
                        <td className="text-right p-2">{trade.shares.toFixed(2)}</td>
                        <td className="text-right p-2 text-orange-500">
                          {trade.slippage_cost !== undefined ? `$${trade.slippage_cost.toFixed(2)}` : "-"}
                        </td>
                        <td className={`text-right p-2 ${trade.pnl && trade.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {trade.pnl !== undefined ? `$${trade.pnl.toFixed(2)}` : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}


