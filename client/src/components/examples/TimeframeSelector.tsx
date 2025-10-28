import { useState } from "react";
import TimeframeSelector, { Timeframe } from "../TimeframeSelector";

export default function TimeframeSelectorExample() {
  const [timeframe, setTimeframe] = useState<Timeframe>("1D");

  return (
    <div className="p-6">
      <TimeframeSelector
        selectedTimeframe={timeframe}
        onSelectTimeframe={(tf) => {
          setTimeframe(tf);
          console.log("Selected timeframe:", tf);
        }}
      />
    </div>
  );
}
