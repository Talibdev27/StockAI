import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { Timeframe } from "./TimeframeSelector";

interface PriceChartProps {
  symbol: string;
  timeframe: Timeframe;
  data: Array<{
    date: string;
    price: number;
    predicted?: boolean;
  }>;
}

export default function PriceChart({ symbol, timeframe, data }: PriceChartProps) {
  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold">Price Chart</h3>
          <p className="text-sm text-muted-foreground">
            {symbol} - {timeframe} Timeframe
          </p>
        </div>

        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(214, 84%, 56%)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="hsl(214, 84%, 56%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis
                dataKey="date"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fontSize: 12 }}
                tickLine={false}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                tick={{ fontSize: 12 }}
                tickLine={false}
                domain={["dataMin - 5", "dataMax + 5"]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Area
                type="monotone"
                dataKey="price"
                stroke="hsl(214, 84%, 56%)"
                strokeWidth={2}
                fill="url(#colorPrice)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[hsl(214,84%,56%)]" />
            <span className="text-muted-foreground">Historical</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-bullish" />
            <span className="text-muted-foreground">Predicted</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
