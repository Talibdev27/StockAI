import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface ModelBreakdownDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelBreakdown?: Record<string, any>;
  predictedPrice?: number;
  actualPrice?: number;
  errorPercent?: number;
  timestamp?: string;
}

export default function ModelBreakdownDialog({
  open,
  onOpenChange,
  modelBreakdown,
  predictedPrice,
  actualPrice,
  errorPercent,
  timestamp,
}: ModelBreakdownDialogProps) {
  if (!modelBreakdown) {
    return null;
  }

  const models = Object.entries(modelBreakdown).map(([name, data]) => ({
    name,
    prediction: (data as any).prediction,
    weight: (data as any).weight,
    confidence: (data as any).confidence,
    performance: (data as any).performance,
  }));

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatPercent = (value: number) => `${value.toFixed(1)}%`;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Model Breakdown</DialogTitle>
          <DialogDescription>
            Individual model predictions and their contributions to the ensemble
            {timestamp && (
              <span className="block mt-1 text-xs text-muted-foreground">
                Prediction made: {new Date(timestamp).toLocaleString()}
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {predictedPrice !== undefined && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <p className="text-sm text-muted-foreground">Predicted Price</p>
                <p className="text-lg font-semibold">{formatPrice(predictedPrice)}</p>
              </div>
              {actualPrice !== undefined && (
                <>
                  <div>
                    <p className="text-sm text-muted-foreground">Actual Price</p>
                    <p className="text-lg font-semibold">{formatPrice(actualPrice)}</p>
                  </div>
                  {errorPercent !== undefined && (
                    <div className="col-span-2">
                      <p className="text-sm text-muted-foreground">Error</p>
                      <p
                        className={`text-lg font-semibold ${
                          errorPercent <= 2
                            ? "text-green-500"
                            : errorPercent > 5
                            ? "text-red-500"
                            : "text-yellow-500"
                        }`}
                      >
                        {formatPercent(errorPercent)}
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          <div>
            <h4 className="text-sm font-semibold mb-2">Individual Model Predictions</h4>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Prediction</TableHead>
                  <TableHead>Weight</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Performance</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.name}>
                    <TableCell className="font-medium capitalize">{model.name}</TableCell>
                    <TableCell>{formatPrice(model.prediction)}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{formatPercent((model.weight || 0) * 100)}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{formatPercent((model.confidence || 0) * 100)}</Badge>
                    </TableCell>
                    <TableCell>
                      {model.performance ? (
                        <div className="text-xs space-y-1">
                          <div>RMSE: {model.performance.rmse?.toFixed(2)}</div>
                          <div>Dir: {formatPercent(model.performance.direction_accuracy || 0)}</div>
                        </div>
                      ) : (
                        <span className="text-muted-foreground text-xs">N/A</span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

