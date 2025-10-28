import { useState } from "react";
import StockSelector from "@/components/StockSelector";
import TimeframeSelector, { Timeframe } from "@/components/TimeframeSelector";
import PriceChart from "@/components/PriceChart";
import PredictionSummary from "@/components/PredictionSummary";
import TechnicalIndicators from "@/components/TechnicalIndicators";
import PerformanceMetrics from "@/components/PerformanceMetrics";
import Watchlist from "@/components/Watchlist";
import RiskDisclaimer from "@/components/RiskDisclaimer";
import { Activity, BarChart3, TrendingUp, Move, Gauge, Target, Award, DollarSign } from "lucide-react";

export default function Dashboard() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [searchQuery, setSearchQuery] = useState("");
  const [timeframe, setTimeframe] = useState<Timeframe>("1D");

  const mockChartData = [
    { date: "Day 1", price: 150 },
    { date: "Day 2", price: 152 },
    { date: "Day 3", price: 148 },
    { date: "Day 4", price: 155 },
    { date: "Day 5", price: 157 },
    { date: "Day 6", price: 160 },
    { date: "Day 7", price: 162 },
    { date: "Day 8", price: 165, predicted: true },
    { date: "Day 9", price: 168, predicted: true },
    { date: "Day 10", price: 170, predicted: true },
  ];

  const mockIndicators = [
    {
      name: "RSI",
      value: 62.5,
      max: 100,
      icon: <Activity className="h-5 w-5 text-primary" />,
      description: "Relative Strength Index",
    },
    {
      name: "MACD",
      value: "+2.45",
      icon: <TrendingUp className="h-5 w-5 text-primary" />,
      description: "Moving Average Convergence",
    },
    {
      name: "MA 50-Day",
      value: "$158.23",
      icon: <Move className="h-5 w-5 text-primary" />,
      description: "50-Day Moving Average",
    },
    {
      name: "MA 200-Day",
      value: "$145.67",
      icon: <Move className="h-5 w-5 text-primary" />,
      description: "200-Day Moving Average",
    },
    {
      name: "Volume",
      value: "42.5M",
      icon: <BarChart3 className="h-5 w-5 text-primary" />,
      description: "Trading Volume",
    },
    {
      name: "Bollinger Bands",
      value: "$165.20",
      icon: <Gauge className="h-5 w-5 text-primary" />,
      description: "Upper Band Price",
    },
  ];

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
                  data={mockChartData}
                />
              </div>
              <div>
                <PredictionSummary
                  currentPrice={162.45}
                  predictedPrice={168.32}
                  confidence={85}
                  signal="BUY"
                />
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Technical Indicators</h2>
              <TechnicalIndicators indicators={mockIndicators} />
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
              <PerformanceMetrics metrics={mockMetrics} />
            </div>

            <RiskDisclaimer />
          </div>

          <div className="lg:w-80">
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
