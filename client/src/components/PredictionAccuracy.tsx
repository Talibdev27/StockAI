import { usePredictionStats } from "@/hooks/useData";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Loader2, TrendingUp, AlertCircle } from "lucide-react";

interface PredictionAccuracyProps {
  symbol: string;
  interval?: string;
}

export default function PredictionAccuracy({ symbol, interval }: PredictionAccuracyProps) {
  const { data: stats, isLoading } = usePredictionStats(symbol, interval);
  
  if (!symbol) {
    return null;
  }
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (!stats || stats.total === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
          <CardDescription>Track prediction performance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-6 text-center">
            <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">No evaluated predictions yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Make predictions and evaluate them to see accuracy metrics
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  const accuracy = stats.direction_accuracy || 0;
  const avgError = stats.avg_error_percent || 0;
  
  // Determine trend (simplified - would need historical comparison)
  const getTrendColor = (val: number) => {
    if (val >= 70) return "text-green-500";
    if (val >= 50) return "text-yellow-500";
    return "text-red-500";
  };
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
        <CardDescription>
          Based on {stats.total} evaluated predictions
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Direction Accuracy */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Direction Accuracy</span>
            <span className={`font-semibold ${getTrendColor(accuracy)}`}>
              {accuracy.toFixed(1)}%
            </span>
          </div>
          <Progress value={accuracy} className="h-2" />
          <p className="text-xs text-muted-foreground">
            Percentage of correct Up/Down/Neutral predictions
          </p>
        </div>
        
        {/* Error Metrics */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t">
          <div>
            <div className="text-xs text-muted-foreground mb-1">Avg Error</div>
            <div className="text-lg font-semibold">
              {avgError.toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              ${stats.rmse?.toFixed(2) || "0.00"} RMSE
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Mean Abs Error</div>
            <div className="text-lg font-semibold">
              ${stats.mae?.toFixed(2) || "0.00"}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Average price difference
            </div>
          </div>
        </div>
        
        {/* Quick Stats */}
        <div className="flex items-center gap-2 pt-2 border-t">
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {accuracy >= 70 ? "Excellent" : accuracy >= 50 ? "Good" : "Needs Improvement"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

