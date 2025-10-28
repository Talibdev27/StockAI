import PredictionSummary from "../PredictionSummary";

export default function PredictionSummaryExample() {
  return (
    <div className="p-6">
      <PredictionSummary
        currentPrice={162.45}
        predictedPrice={168.32}
        confidence={85}
        signal="BUY"
      />
    </div>
  );
}
