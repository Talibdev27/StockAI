import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUp, ArrowDown, TrendingUp } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface PredictionSummaryProps {
  currentPrice: number;
  predictedPrice: number;
  confidence: number;
  signal: "BUY" | "HOLD" | "SELL";
}

export default function PredictionSummary({
  currentPrice,
  predictedPrice,
  confidence,
  signal,
}: PredictionSummaryProps) {
  const change = predictedPrice - currentPrice;
  const changePercent = ((change / currentPrice) * 100).toFixed(2);
  const isPositive = change >= 0;

  const signalColors = {
    BUY: "bg-bullish text-white border-bullish",
    HOLD: "bg-yellow-500 text-white border-yellow-600",
    SELL: "bg-bearish text-white border-bearish",
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold">Prediction Summary</h3>
            <p className="text-sm text-muted-foreground">Next day forecast</p>
          </div>
          <Badge
            className={`${signalColors[signal]} font-bold px-4 py-1 text-sm`}
            data-testid={`badge-signal-${signal.toLowerCase()}`}
          >
            {signal}
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-muted-foreground mb-1">Current Price</p>
            <p className="text-3xl font-mono font-bold" data-testid="text-current-price">
              ${currentPrice.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Predicted Price</p>
            <p className="text-3xl font-mono font-bold text-primary" data-testid="text-predicted-price">
              ${predictedPrice.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 p-4 rounded-md bg-accent/50">
          {isPositive ? (
            <ArrowUp className="h-6 w-6 text-bullish" />
          ) : (
            <ArrowDown className="h-6 w-6 text-bearish" />
          )}
          <div>
            <p className="text-sm text-muted-foreground">Change</p>
            <p
              className={`text-2xl font-mono font-bold ${isPositive ? "text-bullish" : "text-bearish"}`}
              data-testid="text-change-percent"
            >
              {isPositive ? "+" : ""}
              {changePercent}%
            </p>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Prediction Confidence</p>
            <p className="text-sm font-mono font-semibold" data-testid="text-confidence">
              {confidence}%
            </p>
          </div>
          <Progress value={confidence} className="h-2" />
        </div>
      </div>
    </Card>
  );
}
