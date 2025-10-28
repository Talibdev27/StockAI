import PerformanceMetrics from "../PerformanceMetrics";
import { Target, TrendingUp, Award, DollarSign } from "lucide-react";

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

export default function PerformanceMetricsExample() {
  return (
    <div className="p-6">
      <PerformanceMetrics metrics={mockMetrics} />
    </div>
  );
}
