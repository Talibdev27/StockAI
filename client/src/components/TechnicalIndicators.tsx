import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Activity, BarChart3, TrendingUp, Move } from "lucide-react";

interface Indicator {
  name: string;
  value: number | string;
  max?: number;
  icon: React.ReactNode;
  description: string;
}

interface TechnicalIndicatorsProps {
  indicators: Indicator[];
}

export default function TechnicalIndicators({ indicators }: TechnicalIndicatorsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {indicators.map((indicator, index) => (
        <Card key={index} className="p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-md bg-primary/10">
                {indicator.icon}
              </div>
              {typeof indicator.value === "number" && indicator.max && (
                <span className="text-2xl font-mono font-bold">
                  {indicator.value.toFixed(1)}
                </span>
              )}
            </div>

            <div>
              <h4 className="font-semibold text-sm mb-1">{indicator.name}</h4>
              <p className="text-xs text-muted-foreground">{indicator.description}</p>
            </div>

            {typeof indicator.value === "number" && indicator.max ? (
              <Progress
                value={(indicator.value / indicator.max) * 100}
                className="h-2"
              />
            ) : (
              <p className="text-lg font-mono font-semibold text-primary">
                {indicator.value}
              </p>
            )}
          </div>
        </Card>
      ))}
    </div>
  );
}
