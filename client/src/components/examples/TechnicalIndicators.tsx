import TechnicalIndicators from "../TechnicalIndicators";
import { Activity, BarChart3, TrendingUp, Move, Gauge, DollarSign } from "lucide-react";

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

export default function TechnicalIndicatorsExample() {
  return (
    <div className="p-6">
      <TechnicalIndicators indicators={mockIndicators} />
    </div>
  );
}
