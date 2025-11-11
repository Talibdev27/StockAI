import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUp, ArrowDown, TrendingUp, ChevronDown, ChevronUp } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useState } from "react";

interface Model {
  prediction: number;
  confidence: number;
  weight: number;
}

interface PredictionSummaryProps {
  currentPrice: number;
  predictedPrice: number;
  confidence: number;
  signal: "BUY" | "HOLD" | "SELL";
  change?: number; // live change absolute
  changePercent?: number; // live change percent (e.g., 0.23 for 0.23%)
  models?: {
    linear?: Model;
    lstm?: Model;
    arima?: Model;
  };
}

export default function PredictionSummary({
  currentPrice,
  predictedPrice,
  confidence,
  signal,
  change,
  changePercent,
  models,
}: PredictionSummaryProps) {
  const [showModels, setShowModels] = useState(false);
  // Prefer live quote change for intraday. Fallback to prediction delta if not provided.
  const fallbackDelta = predictedPrice - currentPrice;
  const effectiveChange = typeof change === "number" ? change : fallbackDelta;
  const effectiveChangePct =
    typeof changePercent === "number"
      ? changePercent
      : (fallbackDelta / (currentPrice || 1)) * 100;
  const isPositive = effectiveChange >= 0;

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
              {effectiveChangePct.toFixed(2)}%
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

        {models && (
          <div className="border-t pt-4">
            <button
              onClick={() => setShowModels(!showModels)}
              className="flex items-center justify-between w-full text-left hover:bg-accent/50 p-2 rounded-md transition-colors"
            >
              <span className="text-sm font-medium text-muted-foreground">
                Ensemble Model Breakdown
              </span>
              {showModels ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </button>
            
            {showModels && (
              <div className="mt-4 space-y-3">
                {Object.entries(models).map(([name, model]) => (
                  <div key={name} className="flex items-center justify-between p-3 bg-accent/30 rounded-md">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="text-xs font-mono">
                        {name.toUpperCase()}
                      </Badge>
                      <span className="text-sm font-mono">
                        ${model.prediction.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>Conf: {model.confidence}%</span>
                      <span>Weight: {model.weight}%</span>
                    </div>
                  </div>
                ))}
                <div className="text-xs text-muted-foreground mt-2 p-2 bg-accent/20 rounded">
                  <strong>How it works:</strong> Each model makes its own prediction, then we combine them using weighted voting based on their individual confidence scores.
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
