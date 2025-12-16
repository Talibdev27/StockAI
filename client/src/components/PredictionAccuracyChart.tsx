import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, ColorType, LineStyle } from "lightweight-charts";
import { PredictionAccuracyPoint } from "@/lib/predictionAccuracyUtils";
import ModelBreakdownDialog from "./ModelBreakdownDialog";

interface PredictionAccuracyChartProps {
  data: PredictionAccuracyPoint[];
  symbol: string;
  onPointClick?: (point: PredictionAccuracyPoint) => void;
}

export default function PredictionAccuracyChart({
  data,
  symbol,
  onPointClick,
}: PredictionAccuracyChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const predictedSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const confidenceAreaRef = useRef<ISeriesApi<"Area"> | null>(null);
  const confidenceLowerRef = useRef<ISeriesApi<"Area"> | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<PredictionAccuracyPoint | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // Clean up existing chart
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 550,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "rgb(226, 232, 240)",
        fontSize: 12,
      },
      grid: {
        vertLines: { 
          color: "rgba(148, 163, 184, 0.15)",
          style: 1, // Solid lines
        },
        horzLines: { 
          color: "rgba(148, 163, 184, 0.15)",
          style: 1, // Solid lines
        },
      },
      rightPriceScale: {
        borderColor: "rgba(148, 163, 184, 0.3)",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: "rgba(148, 163, 184, 0.3)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1, // Normal mode
        vertLine: {
          color: "rgba(148, 163, 184, 0.5)",
          width: 1,
          style: 2, // Dashed
        },
        horzLine: {
          color: "rgba(148, 163, 184, 0.5)",
          width: 1,
          style: 2, // Dashed
        },
      },
    });

    chartRef.current = chart;

    // Add confidence interval visualization using two area series
    // Upper bound area
    const confidenceUpper = chart.addAreaSeries({
      lineColor: "rgba(148, 163, 184, 0.6)",
      topColor: "rgba(148, 163, 184, 0.25)",
      bottomColor: "rgba(148, 163, 184, 0.05)",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
      priceLineVisible: false,
      lastValueVisible: false,
      lineWidth: 1,
    });
    
    // Lower bound area (will be inverted)
    const confidenceLower = chart.addAreaSeries({
      lineColor: "rgba(148, 163, 184, 0.6)",
      topColor: "rgba(148, 163, 184, 0.05)",
      bottomColor: "rgba(148, 163, 184, 0.25)",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
      priceLineVisible: false,
      lastValueVisible: false,
      lineWidth: 1,
    });
    
    confidenceAreaRef.current = confidenceUpper;
    confidenceLowerRef.current = confidenceLower;

    // Add actual prices line (solid blue) - thicker and more prominent
    const actualSeries = chart.addLineSeries({
      color: "#60a5fa", // Lighter, more vibrant blue
      lineWidth: 3,
      title: "Actual Price",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 5,
    });
    actualSeriesRef.current = actualSeries;

    // Add predicted prices line (dashed orange) - more prominent
    const predictedSeries = chart.addLineSeries({
      color: "#fb923c", // Lighter, more vibrant orange
      lineWidth: 3,
      lineStyle: LineStyle.Dashed,
      title: "Predicted Price",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 5,
    });
    predictedSeriesRef.current = predictedSeries;

    // Helper function to format date to yyyy-mm-dd
    const formatDate = (dateStr: string): string => {
      // If date includes time (ISO format), extract just the date part
      if (dateStr.includes('T')) {
        return dateStr.split('T')[0];
      }
      // If already in yyyy-mm-dd format, return as is
      return dateStr;
    };

    // Sort data by date to ensure ascending order
    const sortedData = [...data].sort((a, b) => {
      const dateA = new Date(a.date).getTime();
      const dateB = new Date(b.date).getTime();
      return dateA - dateB;
    });

    // Prepare data and remove duplicates by keeping the last occurrence
    const dataMap = new Map<string, { actual: number | null; predicted: number; confidence: { upper: number; lower: number } }>();
    
    for (const point of sortedData) {
      const formattedDate = formatDate(point.date);
      dataMap.set(formattedDate, {
        actual: point.actualPrice,
        predicted: point.predictedPrice,
        confidence: point.confidenceInterval,
      });
    }

    // Convert map to arrays, sorted by date
    const uniqueDates = Array.from(dataMap.keys()).sort();
    
    const actualData = uniqueDates
      .map((date) => {
        const point = dataMap.get(date)!;
        return point.actual !== null ? { time: date as any, value: point.actual } : null;
      })
      .filter((item): item is { time: any; value: number } => item !== null);

    const predictedData = uniqueDates.map((date) => {
      const point = dataMap.get(date)!;
      return {
        time: date as any,
        value: point.predicted,
      };
    });

    // For confidence interval, create upper and lower bound data
    const confidenceUpperData = uniqueDates.map((date) => {
      const point = dataMap.get(date)!;
      return {
        time: date as any,
        value: point.confidence.upper,
      };
    });

    const confidenceLowerData = uniqueDates.map((date) => {
      const point = dataMap.get(date)!;
      return {
        time: date as any,
        value: point.confidence.lower,
      };
    });

    // Set data - set confidence areas first (so they appear behind the lines)
    if (confidenceAreaRef.current) {
      confidenceAreaRef.current.setData(confidenceUpperData);
    }
    if (confidenceLowerRef.current) {
      confidenceLowerRef.current.setData(confidenceLowerData);
    }
    
    // Then set the main series
    actualSeries.setData(actualData);
    predictedSeries.setData(predictedData);

    // Add markers for accuracy (using unique dates)
    const markers = uniqueDates
      .map((date) => {
        // Find the corresponding point from sorted data
        const point = sortedData.find((p) => formatDate(p.date) === date);
        if (!point) return null;

        if (point.accuracyStatus === "pending") {
          return {
            time: date as any,
            position: "belowBar" as const,
            color: "#fbbf24", // Brighter yellow
            shape: "circle" as const,
            size: 1.2, // Larger marker
            text: "Pending",
          };
        } else if (point.actualPrice !== null) {
          const color =
            point.accuracyStatus === "accurate"
              ? "#22c55e" // Brighter green
              : point.accuracyStatus === "moderate"
              ? "#fbbf24" // Brighter yellow
              : "#f87171"; // Softer red

          return {
            time: date as any,
            position: "belowBar" as const,
            color,
            shape: "circle" as const,
            size: 1.5, // Larger, more visible markers
            text: point.errorPercent !== null ? `${point.errorPercent.toFixed(1)}%` : "",
          };
        }
        return null;
      })
      .filter((m): m is NonNullable<typeof m> => m !== null);

    predictedSeries.setMarkers(markers as any);

    // Add click handler
    chart.subscribeClick((param) => {
      if (param.point === undefined || param.time === undefined) return;

      // Find the closest data point
      const clickedTime = param.time as string;
      // Format the clicked time to match our date format
      const formattedClickedTime = formatDate(clickedTime);
      const point = sortedData.find((p) => formatDate(p.date) === formattedClickedTime);
      if (point) {
        setSelectedPoint(point);
        setDialogOpen(true);
        if (onPointClick) {
          onPointClick(point);
        }
      }
    });

    // Add crosshair move handler for tooltips
    chart.subscribeCrosshairMove((param) => {
      // Tooltip functionality can be added here if needed
      // For now, we'll rely on the click handler for model breakdown
    });

    chart.timeScale().fitContent();

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
  }, [data, symbol, onPointClick]);

  // Handle confidence interval area updates
  useEffect(() => {
    if (!confidenceAreaRef.current || !confidenceLowerRef.current || !data || data.length === 0) return;

    // Helper function to format date to yyyy-mm-dd
    const formatDate = (dateStr: string): string => {
      if (dateStr.includes('T')) {
        return dateStr.split('T')[0];
      }
      return dateStr;
    };

    // Sort data by date to ensure ascending order
    const sortedData = [...data].sort((a, b) => {
      const dateA = new Date(a.date).getTime();
      const dateB = new Date(b.date).getTime();
      return dateA - dateB;
    });

    // Deduplicate by date (keep last occurrence)
    const dataMap = new Map<string, { confidence: { upper: number; lower: number } }>();
    
    for (const point of sortedData) {
      const formattedDate = formatDate(point.date);
      dataMap.set(formattedDate, {
        confidence: point.confidenceInterval,
      });
    }

    // Convert map to array, sorted by date
    const uniqueDates = Array.from(dataMap.keys()).sort();

    // Create upper and lower bound data
    const confidenceUpperData = uniqueDates.map((date) => {
      const point = dataMap.get(date)!;
      return {
        time: date as any,
        value: point.confidence.upper,
      };
    });

    const confidenceLowerData = uniqueDates.map((date) => {
      const point = dataMap.get(date)!;
      return {
        time: date as any,
        value: point.confidence.lower,
      };
    });

    confidenceAreaRef.current.setData(confidenceUpperData);
    confidenceLowerRef.current.setData(confidenceLowerData);
  }, [data]);

  return (
    <>
      <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden" style={{ height: "550px" }} />
      {selectedPoint && (
        <ModelBreakdownDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          modelBreakdown={selectedPoint.modelBreakdown}
          predictedPrice={selectedPoint.predictedPrice}
          actualPrice={selectedPoint.actualPrice ?? undefined}
          errorPercent={selectedPoint.errorPercent ?? undefined}
          timestamp={selectedPoint.date}
        />
      )}
    </>
  );
}

