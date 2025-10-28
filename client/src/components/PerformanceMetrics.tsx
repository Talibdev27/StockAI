import { Card } from "@/components/ui/card";
import { Target, TrendingUp, Award, DollarSign } from "lucide-react";

interface Metric {
  name: string;
  value: string;
  icon: React.ReactNode;
  description: string;
}

interface PerformanceMetricsProps {
  metrics: Metric[];
}

export default function PerformanceMetrics({ metrics }: PerformanceMetricsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => (
        <Card key={index} className="p-6">
          <div className="space-y-4">
            <div className="p-3 rounded-md bg-primary/10 w-fit">
              {metric.icon}
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">{metric.name}</p>
              <p className="text-3xl font-mono font-bold" data-testid={`text-metric-${metric.name.toLowerCase().replace(" ", "-")}`}>
                {metric.value}
              </p>
            </div>
            <p className="text-xs text-muted-foreground">{metric.description}</p>
          </div>
        </Card>
      ))}
    </div>
  );
}
