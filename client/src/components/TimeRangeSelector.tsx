import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

export type TimeRange = "7d" | "30d" | "90d" | "custom";

export interface DateRange {
  from: Date | undefined;
  to: Date | undefined;
}

interface TimeRangeSelectorProps {
  selectedRange: TimeRange;
  onRangeChange: (range: TimeRange) => void;
  customDateRange?: DateRange;
  onCustomDateRangeChange?: (range: DateRange) => void;
}

export default function TimeRangeSelector({
  selectedRange,
  onRangeChange,
  customDateRange,
  onCustomDateRangeChange,
}: TimeRangeSelectorProps) {
  const getDateFromRange = (range: TimeRange): Date => {
    const now = new Date();
    const days = range === "7d" ? 7 : range === "30d" ? 30 : 90;
    const date = new Date(now);
    date.setDate(date.getDate() - days);
    return date;
  };

  return (
    <div className="flex items-center gap-2">
      <Select value={selectedRange} onValueChange={(value) => onRangeChange(value as TimeRange)}>
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Select range" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="7d">Last 7 days</SelectItem>
          <SelectItem value="30d">Last 30 days</SelectItem>
          <SelectItem value="90d">Last 90 days</SelectItem>
          <SelectItem value="custom">Custom range</SelectItem>
        </SelectContent>
      </Select>

      {selectedRange === "custom" && onCustomDateRangeChange && (
        <div className="flex items-center gap-2">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-[240px] justify-start text-left font-normal",
                  !customDateRange?.from && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {customDateRange?.from ? (
                  format(customDateRange.from, "PPP")
                ) : (
                  <span>Pick a start date</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={customDateRange?.from}
                onSelect={(date) =>
                  onCustomDateRangeChange({
                    from: date,
                    to: customDateRange?.to,
                  })
                }
                initialFocus
              />
            </PopoverContent>
          </Popover>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-[240px] justify-start text-left font-normal",
                  !customDateRange?.to && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {customDateRange?.to ? (
                  format(customDateRange.to, "PPP")
                ) : (
                  <span>Pick an end date</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={customDateRange?.to}
                onSelect={(date) =>
                  onCustomDateRangeChange({
                    from: customDateRange?.from,
                    to: date,
                  })
                }
                initialFocus
              />
            </PopoverContent>
          </Popover>
        </div>
      )}
    </div>
  );
}

