import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { PatternName } from "@/lib/candlestickPatterns";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp, Eye, EyeOff } from "lucide-react";
import { useState } from "react";

interface PatternControlsProps {
  enabledPatterns: Set<PatternName>;
  onTogglePattern: (name: PatternName) => void;
  strengthThreshold: number;
  onStrengthThresholdChange: (value: number) => void;
  compactMode: boolean;
  onCompactModeChange: (value: boolean) => void;
}

const patternLabels: Record<PatternName, string> = {
  Doji: "Doji",
  Hammer: "Hammer",
  "Shooting Star": "Shooting Star",
  "Bullish Engulfing": "Bullish Engulfing",
  "Bearish Engulfing": "Bearish Engulfing",
};

export default function PatternControls({
  enabledPatterns,
  onTogglePattern,
  strengthThreshold,
  onStrengthThresholdChange,
  compactMode,
  onCompactModeChange,
}: PatternControlsProps) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="w-full">
      <CollapsibleTrigger asChild>
        <button className="flex items-center justify-between w-full text-sm text-muted-foreground hover:text-foreground transition-colors">
          <span>Pattern Settings</span>
          {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2">
        <Card className="p-4 space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="compact-mode" className="text-sm">
                Compact Mode
              </Label>
              <Switch
                id="compact-mode"
                checked={compactMode}
                onCheckedChange={onCompactModeChange}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Hide markers on chart, show only in panel
            </p>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">
              Strength Threshold: {strengthThreshold}
            </Label>
            <Slider
              value={[strengthThreshold]}
              onValueChange={([val]) => onStrengthThresholdChange(val)}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Only show patterns with strength ≥ {strengthThreshold}
            </p>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Pattern Types</Label>
            <div className="space-y-2">
              {(Object.keys(patternLabels) as PatternName[]).map((name) => (
                <div key={name} className="flex items-center justify-between">
                  <Label htmlFor={`pattern-${name}`} className="text-xs font-normal">
                    {patternLabels[name]}
                  </Label>
                  <Switch
                    id={`pattern-${name}`}
                    checked={enabledPatterns.has(name)}
                    onCheckedChange={() => onTogglePattern(name)}
                  />
                </div>
              ))}
            </div>
          </div>
        </Card>
      </CollapsibleContent>
    </Collapsible>
  );
}

