import { Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";

interface StockSelectorProps {
  selectedStock: string;
  onSelectStock: (symbol: string) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

const popularStocks = [
  { symbol: "AAPL", name: "Apple" },
  { symbol: "GOOGL", name: "Google" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "TSLA", name: "Tesla" },
  { symbol: "AMZN", name: "Amazon" },
];

export default function StockSelector({
  selectedStock,
  onSelectStock,
  searchQuery,
  onSearchChange,
}: StockSelectorProps) {
  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search stock symbol..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
          data-testid="input-stock-search"
        />
      </div>

      <div className="flex flex-wrap gap-2">
        <span className="text-sm text-muted-foreground self-center">Popular:</span>
        {popularStocks.map((stock) => (
          <Badge
            key={stock.symbol}
            variant={selectedStock === stock.symbol ? "default" : "secondary"}
            className="cursor-pointer hover-elevate active-elevate-2"
            onClick={() => onSelectStock(stock.symbol)}
            data-testid={`badge-stock-${stock.symbol}`}
          >
            {stock.symbol}
          </Badge>
        ))}
      </div>
    </div>
  );
}
