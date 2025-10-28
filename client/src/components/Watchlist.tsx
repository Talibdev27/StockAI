import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LineChart, Line, ResponsiveContainer } from "recharts";
import { Plus, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface WatchlistStock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  sparklineData: number[];
}

interface WatchlistProps {
  stocks: WatchlistStock[];
  onRemove: (symbol: string) => void;
  onAdd: () => void;
}

export default function Watchlist({ stocks, onRemove, onAdd }: WatchlistProps) {
  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Watchlist</h3>
          <Button
            variant="ghost"
            size="icon"
            onClick={onAdd}
            data-testid="button-add-watchlist"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-3">
          {stocks.map((stock) => (
            <div
              key={stock.symbol}
              className="flex items-center gap-3 p-3 rounded-md border hover-elevate active-elevate-2 cursor-pointer"
              data-testid={`card-watchlist-${stock.symbol}`}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <div>
                    <p className="font-semibold text-sm">{stock.symbol}</p>
                    <p className="text-xs text-muted-foreground truncate">
                      {stock.name}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(stock.symbol);
                    }}
                    data-testid={`button-remove-${stock.symbol}`}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <p className="text-sm font-mono font-semibold">
                    ${stock.price.toFixed(2)}
                  </p>
                  <Badge
                    variant="secondary"
                    className={`text-xs ${stock.change >= 0 ? "text-bullish" : "text-bearish"}`}
                  >
                    {stock.change >= 0 ? "+" : ""}
                    {stock.changePercent.toFixed(2)}%
                  </Badge>
                </div>
              </div>
              <div className="w-16 h-10">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stock.sparklineData.map((value, i) => ({ value, index: i }))}>
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={stock.change >= 0 ? "rgb(16, 185, 129)" : "rgb(239, 68, 68)"}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
