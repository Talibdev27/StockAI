import Watchlist from "../Watchlist";

const mockStocks = [
  {
    symbol: "AAPL",
    name: "Apple Inc.",
    price: 162.45,
    change: 3.25,
    changePercent: 2.04,
    sparklineData: [158, 159, 157, 160, 162, 161, 162.45],
  },
  {
    symbol: "GOOGL",
    name: "Alphabet Inc.",
    price: 138.72,
    change: -1.28,
    changePercent: -0.91,
    sparklineData: [142, 140, 141, 139, 138, 139, 138.72],
  },
  {
    symbol: "MSFT",
    name: "Microsoft Corp.",
    price: 378.91,
    change: 5.67,
    changePercent: 1.52,
    sparklineData: [372, 375, 373, 376, 379, 377, 378.91],
  },
];

export default function WatchlistExample() {
  return (
    <div className="p-6">
      <Watchlist
        stocks={mockStocks}
        onRemove={(symbol) => console.log("Remove:", symbol)}
        onAdd={() => console.log("Add new stock")}
      />
    </div>
  );
}
