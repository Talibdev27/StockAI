import { useMemo } from "react";
import { Search, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { useStocks } from "@/hooks/useData";

interface StockSelectorProps {
  selectedStock: string;
  onSelectStock: (symbol: string) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

interface Stock {
  symbol: string;
  name: string;
}

export default function StockSelector({
  selectedStock,
  onSelectStock,
  searchQuery,
  onSearchChange,
}: StockSelectorProps) {
  const { data: allStocks, isLoading, error } = useStocks();
  
  // Filter stocks based on search query
  const filteredStocks = useMemo(() => {
    if (!allStocks || !searchQuery.trim()) {
      return [];
    }
    
    const query = searchQuery.toLowerCase().trim();
    return allStocks.filter(
      (stock: Stock) =>
        stock.symbol.toLowerCase().includes(query) ||
        stock.name.toLowerCase().includes(query)
    ).slice(0, 10); // Limit to 10 results for performance
  }, [allStocks, searchQuery]);

  // Get popular stocks (fetch with popular=true param)
  const popularList = useMemo(() => {
    if (!allStocks) return [];
    // Top 10 most popular stocks by market cap
    const popularSymbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "V", "JNJ"];
    return allStocks.filter((s: Stock) => popularSymbols.includes(s.symbol))
      .sort((a: Stock, b: Stock) => popularSymbols.indexOf(a.symbol) - popularSymbols.indexOf(b.symbol))
      .slice(0, 10);
  }, [allStocks]);

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search stock symbol or name..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
          data-testid="input-stock-search"
        />
        
        {/* Search Results Dropdown */}
        {searchQuery.trim() && (
          <Card className="absolute z-50 w-full mt-1 shadow-lg">
            {isLoading ? (
              <div className="p-3 flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                <div className="text-sm text-muted-foreground">Loading stocks...</div>
              </div>
            ) : error ? (
              <div className="p-3">
                <div className="text-sm text-destructive font-medium">Failed to load stocks</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {error instanceof Error ? error.message : "Please try again later"}
                </div>
              </div>
            ) : filteredStocks.length > 0 ? (
              <div className="max-h-60 overflow-y-auto">
                <div className="p-2">
                  {filteredStocks.map((stock: Stock) => (
                    <div
                      key={stock.symbol}
                      className="p-2 hover:bg-accent rounded cursor-pointer flex items-center justify-between"
                      onClick={() => {
                        onSelectStock(stock.symbol);
                        onSearchChange("");
                      }}
                    >
                      <div>
                        <div className="font-medium">{stock.symbol}</div>
                        <div className="text-xs text-muted-foreground">{stock.name}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : allStocks && allStocks.length > 0 ? (
              <div className="p-3">
                <div className="text-sm text-muted-foreground">No stocks found matching "{searchQuery}"</div>
              </div>
            ) : null}
          </Card>
        )}
      </div>

      {/* Popular Stocks */}
      <div className="flex flex-wrap gap-2">
        <span className="text-sm text-muted-foreground self-center">Popular:</span>
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        ) : (
          (popularList.length > 0 ? popularList : [
            { symbol: "AAPL", name: "Apple Inc." },
            { symbol: "MSFT", name: "Microsoft Corp." },
            { symbol: "GOOGL", name: "Alphabet Inc." },
            { symbol: "AMZN", name: "Amazon.com Inc." },
            { symbol: "NVDA", name: "NVIDIA Corporation" },
          ]).slice(0, 10).map((stock: Stock) => (
            <Badge
              key={stock.symbol}
              variant={selectedStock === stock.symbol ? "default" : "secondary"}
              className="cursor-pointer hover:bg-primary/80 transition-colors"
              onClick={() => onSelectStock(stock.symbol)}
              data-testid={`badge-stock-${stock.symbol}`}
            >
              {stock.symbol}
            </Badge>
          ))
        )}
      </div>
    </div>
  );
}
