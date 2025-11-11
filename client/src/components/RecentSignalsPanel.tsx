import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Pin, PinOff, X, Bell } from "lucide-react";
import { DetectedPattern, PatternName } from "@/lib/candlestickPatterns";

interface RecentSignalsPanelProps {
  signals: DetectedPattern[];
  pinnedSignals: Set<number>;
  onPin: (index: number) => void;
  onUnpin: (index: number) => void;
  onHideType: (name: PatternName) => void;
  onCreateAlert: (signal: DetectedPattern) => void;
}

const directionColors = {
  bullish: "text-green-500",
  bearish: "text-red-500",
  neutral: "text-gray-400",
};

const directionIcons = {
  bullish: "🟢",
  bearish: "🔴",
  neutral: "⚪",
};

export default function RecentSignalsPanel({
  signals,
  pinnedSignals,
  onPin,
  onUnpin,
  onHideType,
  onCreateAlert,
}: RecentSignalsPanelProps) {
  // Get last 3 strong signals (sorted by strength, descending)
  const recent = signals
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 3);

  if (recent.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="text-sm font-semibold mb-2">Recent Signals</h3>
        <p className="text-xs text-muted-foreground">No patterns detected</p>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Recent Signals</h3>
        <Badge variant="outline" className="text-xs">
          {signals.length} total
        </Badge>
      </div>
      <div className="space-y-3">
        {recent.map((signal, idx) => {
          const isPinned = pinnedSignals.has(signal.index);
          return (
            <div
              key={`${signal.index}-${idx}`}
              className="p-2 rounded-md bg-accent/30 border border-border"
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs">{directionIcons[signal.direction]}</span>
                  <span className="text-xs font-medium">{signal.name}</span>
                  <Badge
                    variant={signal.strength >= 70 ? "default" : "secondary"}
                    className="text-xs h-5 px-1.5"
                  >
                    {signal.strength}
                  </Badge>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0"
                    onClick={() => (isPinned ? onUnpin(signal.index) : onPin(signal.index))}
                    title={isPinned ? "Unpin" : "Pin"}
                  >
                    {isPinned ? (
                      <PinOff className="h-3 w-3 text-primary" />
                    ) : (
                      <Pin className="h-3 w-3" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0"
                    onClick={() => onHideType(signal.name as PatternName)}
                    title="Hide this pattern type"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mb-1">{signal.rationale}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {new Date(signal.date).toLocaleDateString()}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-6 text-xs"
                  onClick={() => onCreateAlert(signal)}
                >
                  <Bell className="h-3 w-3 mr-1" />
                  Alert
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

