import PriceChart from "../PriceChart";

const mockData = [
  { date: "Jan 1", price: 150 },
  { date: "Jan 2", price: 152 },
  { date: "Jan 3", price: 148 },
  { date: "Jan 4", price: 155 },
  { date: "Jan 5", price: 157 },
  { date: "Jan 6", price: 160 },
  { date: "Jan 7", price: 162 },
  { date: "Jan 8", price: 165, predicted: true },
  { date: "Jan 9", price: 168, predicted: true },
  { date: "Jan 10", price: 170, predicted: true },
];

export default function PriceChartExample() {
  return (
    <div className="p-6">
      <PriceChart symbol="AAPL" timeframe="1D" data={mockData} />
    </div>
  );
}
