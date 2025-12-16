import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Target, TrendingUp, Award, BarChart3, Loader2 } from "lucide-react";
import { PredictionStats } from "@/hooks/useData";

interface ModelPerformance {
  rmse: number;
  mae: number;
  direction_accuracy: number;
  count: number;
  score: number;
}

interface ModelPerformanceMetrics {
  [modelName: string]: ModelPerformance;
}

interface PredictionMetricsCardsProps {
  stats?: PredictionStats;
  modelMetrics?: ModelPerformanceMetrics;
  isLoading?: boolean;
}

export default function PredictionMetricsCards({
  stats,
  modelMetrics,
  isLoading,
}: PredictionMetricsCardsProps) {
  // Find best and worst performing models
  const getBestWorstModels = () => {
    if (!modelMetrics || Object.keys(modelMetrics).length === 0) {
      return { best: null, worst: null };
    }

    const models = Object.entries(modelMetrics);
    const sorted = models.sort((a, b) => b[1].score - a[1].score);
    return {
      best: sorted[0] ? { name: sorted[0][0], ...sorted[0][1] } : null,
      worst: sorted[sorted.length - 1]
        ? { name: sorted[sorted.length - 1][0], ...sorted[sorted.length - 1][1] }
        : null,
    };
  };

  const { best, worst } = getBestWorstModels();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardHeader>
              <CardTitle className="text-sm font-medium">
                <Loader2 className="h-4 w-4 animate-spin inline mr-2" />
                Loading...
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">-</div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Overall Accuracy */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Overall Accuracy</CardTitle>
          <Target className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats?.direction_accuracy?.toFixed(1) || 0}%
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {stats?.total || 0} predictions evaluated
          </p>
        </CardContent>
      </Card>

      {/* Average Prediction Error */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Average Error</CardTitle>
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats?.avg_error_percent?.toFixed(2) || 0}%
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            RMSE: ${stats?.rmse?.toFixed(2) || 0} | MAE: ${stats?.mae?.toFixed(2) || 0}
          </p>
        </CardContent>
      </Card>

      {/* Direction Accuracy */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Direction Accuracy</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats?.direction_accuracy?.toFixed(1) || 0}%
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Correct up/down predictions
          </p>
        </CardContent>
      </Card>

      {/* Best/Worst Model */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Best Model</CardTitle>
          <Award className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {best ? (
            <>
              <div className="text-lg font-bold capitalize">{best.name}</div>
              <div className="text-xs text-muted-foreground mt-1 space-y-0.5">
                <div>Accuracy: {best.direction_accuracy.toFixed(1)}%</div>
                <div>RMSE: ${best.rmse.toFixed(2)}</div>
                <div className="text-xs">
                  <Badge variant="secondary" className="text-xs">
                    {best.count} evaluations
                  </Badge>
                </div>
              </div>
            </>
          ) : worst ? (
            <>
              <div className="text-lg font-bold capitalize">{worst.name}</div>
              <div className="text-xs text-muted-foreground mt-1">
                Only model with data
              </div>
            </>
          ) : (
            <div className="text-sm text-muted-foreground">No model data</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

