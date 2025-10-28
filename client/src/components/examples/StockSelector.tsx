import { useState } from "react";
import StockSelector from "../StockSelector";

export default function StockSelectorExample() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <div className="p-6">
      <StockSelector
        selectedStock={selectedStock}
        onSelectStock={(symbol) => {
          setSelectedStock(symbol);
          console.log("Selected stock:", symbol);
        }}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />
    </div>
  );
}
