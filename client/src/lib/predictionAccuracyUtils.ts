import { PredictionHistoryItem } from "@/hooks/useData";

export interface PredictionAccuracyPoint {
  date: string;
  actualPrice: number | null;
  predictedPrice: number;
  confidence: number;
  confidenceInterval: { upper: number; lower: number };
  errorPercent: number | null;
  accuracyStatus: "accurate" | "moderate" | "inaccurate" | "pending";
  modelBreakdown?: Record<string, any>;
}

/**
 * Calculate confidence interval from model breakdown predictions
 */
export function calculateConfidenceInterval(
  predictedPrice: number,
  modelBreakdown?: Record<string, any>
): { upper: number; lower: number } {
  if (!modelBreakdown || Object.keys(modelBreakdown).length === 0) {
    // Default to ±5% if no model breakdown
    return {
      upper: predictedPrice * 1.05,
      lower: predictedPrice * 0.95,
    };
  }

  // Extract individual model predictions
  const modelPredictions = Object.values(modelBreakdown)
    .map((model: any) => model.prediction)
    .filter((pred): pred is number => typeof pred === "number");

  if (modelPredictions.length === 0) {
    return {
      upper: predictedPrice * 1.05,
      lower: predictedPrice * 0.95,
    };
  }

  // Calculate standard deviation
  const mean = modelPredictions.reduce((sum, pred) => sum + pred, 0) / modelPredictions.length;
  const variance =
    modelPredictions.reduce((sum, pred) => sum + Math.pow(pred - mean, 2), 0) /
    modelPredictions.length;
  const stdDev = Math.sqrt(variance);

  // Confidence interval = ±1 standard deviation (68% confidence)
  return {
    upper: predictedPrice + stdDev,
    lower: predictedPrice - stdDev,
  };
}

/**
 * Classify prediction accuracy based on error percentage
 */
export function classifyAccuracy(
  errorPercent: number | null,
  actualPrice: number | null
): "accurate" | "moderate" | "inaccurate" | "pending" {
  if (actualPrice === null || errorPercent === null) {
    return "pending";
  }

  if (errorPercent <= 2) {
    return "accurate";
  } else if (errorPercent <= 5) {
    return "moderate";
  } else {
    return "inaccurate";
  }
}

/**
 * Transform prediction history item to accuracy point
 */
export function transformHistoryItemToAccuracyPoint(
  item: PredictionHistoryItem
): PredictionAccuracyPoint {
  const errorPercent =
    item.actual_price !== undefined && item.error_percent !== undefined
      ? item.error_percent
      : null;

  const confidenceInterval = calculateConfidenceInterval(
    item.predicted_price,
    item.model_breakdown
  );

  const accuracyStatus = classifyAccuracy(errorPercent, item.actual_price ?? null);

  return {
    date: item.timestamp,
    actualPrice: item.actual_price ?? null,
    predictedPrice: item.predicted_price,
    confidence: item.confidence,
    confidenceInterval,
    errorPercent,
    accuracyStatus,
    modelBreakdown: item.model_breakdown,
  };
}

/**
 * Filter data points by time range
 */
export function filterByTimeRange(
  points: PredictionAccuracyPoint[],
  range: "7d" | "30d" | "90d" | "custom",
  customRange?: { from: Date; to: Date }
): PredictionAccuracyPoint[] {
  if (range === "custom" && customRange) {
    const fromTime = customRange.from.getTime();
    const toTime = customRange.to.getTime();
    return points.filter((point) => {
      const pointTime = new Date(point.date).getTime();
      return pointTime >= fromTime && pointTime <= toTime;
    });
  }

  const now = new Date();
  const days = range === "7d" ? 7 : range === "30d" ? 30 : 90;
  const cutoffDate = new Date(now);
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return points.filter((point) => {
    const pointDate = new Date(point.date);
    return pointDate >= cutoffDate;
  });
}

/**
 * Calculate direction accuracy from prediction points
 */
export function calculateDirectionAccuracy(
  points: PredictionAccuracyPoint[]
): number {
  const evaluatedPoints = points.filter(
    (p) => p.actualPrice !== null && p.errorPercent !== null
  );

  if (evaluatedPoints.length === 0) {
    return 0;
  }

  let correct = 0;
  for (const point of evaluatedPoints) {
    if (point.actualPrice === null || point.predictedPrice === undefined) continue;

    const predictedDirection = point.predictedPrice > point.actualPrice ? "up" : "down";
    // We need to compare with previous actual price to determine actual direction
    // For now, we'll use a simplified approach if we have the data
    // This would need actual price history to be fully accurate
  }

  // For now, return 0 - this would need enhancement with actual price history
  // The backend should provide direction accuracy in stats
  return 0;
}

/**
 * Calculate overall accuracy metrics from points
 */
export function calculateAccuracyMetrics(points: PredictionAccuracyPoint[]): {
  overallAccuracy: number;
  avgError: number;
  rmse: number;
  mae: number;
  directionAccuracy: number;
} {
  const evaluatedPoints = points.filter(
    (p) => p.actualPrice !== null && p.errorPercent !== null
  );

  if (evaluatedPoints.length === 0) {
    return {
      overallAccuracy: 0,
      avgError: 0,
      rmse: 0,
      mae: 0,
      directionAccuracy: 0,
    };
  }

  const errors = evaluatedPoints
    .map((p) => Math.abs(p.errorPercent ?? 0))
    .filter((e) => !isNaN(e));

  const avgError = errors.reduce((sum, e) => sum + e, 0) / errors.length;

  // Calculate RMSE and MAE from absolute errors
  const squaredErrors = errors.map((e) => e * e);
  const rmse = Math.sqrt(
    squaredErrors.reduce((sum, e) => sum + e, 0) / squaredErrors.length
  );

  const mae = errors.reduce((sum, e) => sum + e, 0) / errors.length;

  // Overall accuracy = percentage of accurate predictions (≤2% error)
  const accurateCount = evaluatedPoints.filter(
    (p) => p.accuracyStatus === "accurate"
  ).length;
  const overallAccuracy = (accurateCount / evaluatedPoints.length) * 100;

  return {
    overallAccuracy,
    avgError,
    rmse,
    mae,
    directionAccuracy: 0, // Would need direction data
  };
}

