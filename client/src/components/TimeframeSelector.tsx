import { Button } from "@/components/ui/button";

export type Timeframe = "1H" | "4H" | "1D" | "1W" | "1M";

interface TimeframeSelectorProps {
  selectedTimeframe: Timeframe;
  onSelectTimeframe: (timeframe: Timeframe) => void;
}

const timeframes: Timeframe[] = ["1H", "4H", "1D", "1W", "1M"];

export default function TimeframeSelector({
  selectedTimeframe,
  onSelectTimeframe,
}: TimeframeSelectorProps) {
  return (
    <div className="flex gap-1">
      {timeframes.map((tf) => (
        <Button
          key={tf}
          variant={selectedTimeframe === tf ? "default" : "ghost"}
          size="sm"
          onClick={() => onSelectTimeframe(tf)}
          className={selectedTimeframe === tf ? "font-mono" : "font-mono text-muted-foreground"}
          data-testid={`button-timeframe-${tf}`}
        >
          {tf}
        </Button>
      ))}
    </div>
  );
}
