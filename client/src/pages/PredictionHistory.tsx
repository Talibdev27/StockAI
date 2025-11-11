import { useState, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { usePredictionHistory, usePredictionStats, useEvaluation, PredictionHistoryItem } from "@/hooks/useData";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, TrendingDown, Minus, RefreshCw, CheckCircle2, XCircle, Zap, Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import StockSelector from "@/components/StockSelector";
import HelpSection from "@/components/HelpSection";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { apiDelete } from "@/lib/api";

export default function PredictionHistory() {
  const [symbol, setSymbol] = useState<string | undefined>("AAPL");
  const [viewMode, setViewMode] = useState<"single" | "all">("single");
  const [searchQuery, setSearchQuery] = useState("");
  const [interval, setInterval] = useState<string>("all");
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [predictionToDelete, setPredictionToDelete] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);
  
  const queryClient = useQueryClient();
  const { toast } = useToast();
  
  // Use viewMode to determine if we show all stocks or just selected
  const historySymbol = viewMode === "all" ? undefined : symbol;
  
  const { data: history, isLoading: historyLoading, refetch: refetchHistory } = usePredictionHistory(
    historySymbol,
    limit,
    0,
    interval === "all" ? undefined : interval
  );
  
  // Refetch when viewMode changes
  useEffect(() => {
    refetchHistory();
  }, [viewMode, refetchHistory]);
  
  // Get unique symbols from history to show if we actually have multiple stocks
  const uniqueSymbols = history ? [...new Set(history.map((item: PredictionHistoryItem) => item.symbol))] : [];
  
  const { data: stats, isLoading: statsLoading } = usePredictionStats(
    viewMode === "all" ? undefined : symbol, 
    interval === "all" ? undefined : interval
  );
  
  const { 
    data: evaluationResult, 
    isLoading: evaluating,
    refetch: triggerEvaluation 
  } = useEvaluation(symbol, 100);
  
  // Separate hook for batch evaluation (all stocks)
  const { 
    data: batchEvaluationResult, 
    isLoading: evaluatingAll,
    refetch: triggerBatchEvaluation 
  } = useEvaluation(undefined, 5000);
  
  const handleEvaluate = async () => {
    try {
      const result = await triggerEvaluation();
      queryClient.invalidateQueries({ queryKey: ["prediction-history"] });
      queryClient.invalidateQueries({ queryKey: ["prediction-stats"] });
      queryClient.invalidateQueries({ queryKey: ["prediction-stats", symbol] });
      
      const evaluated = result.data?.evaluated || 0;
      const message = result.data?.message;
      
      if (evaluated === 0 && message) {
        toast({
          title: "No Predictions to Evaluate",
          description: message,
          variant: "default",
          duration: 5000,
        });
      } else {
        toast({
          title: "Evaluation Complete",
          description: `Evaluated ${evaluated} prediction${evaluated !== 1 ? 's' : ''} for ${symbol}`,
        });
      }
    } catch (error) {
      toast({
        title: "Evaluation Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };
  
  const handleDelete = async (predictionId: number) => {
    setPredictionToDelete(predictionId);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!predictionToDelete) return;

    setDeleting(true);
    try {
      await apiDelete(`/api/predictions/${predictionToDelete}`);
      queryClient.invalidateQueries({ queryKey: ["prediction-history"] });
      queryClient.invalidateQueries({ queryKey: ["prediction-stats"] });
      queryClient.invalidateQueries({ queryKey: ["trading-performance"] });
      toast({
        title: "Prediction Deleted",
        description: "The prediction has been deleted successfully",
      });
      setDeleteDialogOpen(false);
      setPredictionToDelete(null);
    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error instanceof Error ? error.message : "Failed to delete prediction",
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
    }
  };

  const handleRefresh = async () => {
    try {
      await refetchHistory();
      // Also refresh stats
      queryClient.invalidateQueries({ queryKey: ["prediction-stats"] });
      toast({
        title: "Refreshed",
        description: "Prediction history updated",
        duration: 2000,
      });
    } catch (error) {
      toast({
        title: "Refresh Failed",
        description: error instanceof Error ? error.message : "Failed to refresh data",
        variant: "destructive",
      });
    }
  };

  const handleEvaluateAll = async () => {
    try {
      const result = await triggerBatchEvaluation();
      queryClient.invalidateQueries({ queryKey: ["prediction-history"] });
      queryClient.invalidateQueries({ queryKey: ["prediction-stats"] });
      const evaluated = result.data?.evaluated || 0;
      const errors = result.data?.errors || 0;
      const total = result.data?.total || 0;
      toast({
        title: "Batch Evaluation Complete",
        description: `Evaluated ${evaluated} predictions across all stocks${errors > 0 ? ` (${errors} errors)` : ''}`,
        duration: 5000,
      });
    } catch (error) {
      toast({
        title: "Batch Evaluation Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };
  
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  const formatPrice = (price: number) => {
    return `$${price.toFixed(2)}`;
  };
  
  const getDirectionIcon = (direction?: "up" | "down" | "neutral") => {
    if (direction === "up") return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (direction === "down") return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };
  
  const getDirectionBadge = (direction?: "up" | "down" | "neutral") => {
    if (direction === "up") return <Badge variant="outline" className="text-green-500 border-green-500">Up</Badge>;
    if (direction === "down") return <Badge variant="outline" className="text-red-500 border-red-500">Down</Badge>;
    return <Badge variant="outline">Neutral</Badge>;
  };

  const handleSelectStock = (selectedSymbol: string) => {
    setSymbol(selectedSymbol);
    setSearchQuery("");
  };
  
  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Prediction History</h1>
      </div>
      
      <HelpSection title="Evaluation Guide">
        <div className="space-y-3">
          <div>
            <p className="font-medium mb-1">1. Viewing Predictions</p>
            <p>Use the <strong>Single Stock</strong> toggle to view predictions for a specific stock, or <strong>All Stocks</strong> to see predictions across all stocks you've made. Use the search bar to quickly find stocks by symbol or company name.</p>
          </div>
          <div>
            <p className="font-medium mb-1">2. How Evaluation Works</p>
            <p>Evaluation compares your predicted prices with actual market prices. When you click <strong>Evaluate</strong>, the system fetches the current market price and calculates the error, direction accuracy, and whether your prediction was correct.</p>
          </div>
          <div>
            <p className="font-medium mb-1">3. Evaluation Buttons</p>
            <ul className="list-disc list-inside ml-2 space-y-1">
              <li><strong>Evaluate {symbol || "Stock"}</strong>: Evaluates pending predictions for the selected stock only</li>
              <li><strong>Evaluate All Stocks</strong>: Evaluates all pending predictions across all stocks (useful for batch processing)</li>
            </ul>
          </div>
          <div>
            <p className="font-medium mb-1">4. Troubleshooting</p>
            <ul className="list-disc list-inside ml-2 space-y-1">
              <li><strong>"No predictions found"</strong>: Make predictions first from the Dashboard page</li>
              <li><strong>"Evaluated 0 predictions"</strong>: Either no predictions exist for that stock, or all predictions are already evaluated</li>
              <li><strong>Only seeing one stock</strong>: Make predictions for multiple stocks on the Dashboard, then switch to "All Stocks" view</li>
            </ul>
          </div>
        </div>
      </HelpSection>
      
      {/* Filters and Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Filters & Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2 md:col-span-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="symbol">Symbol</Label>
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-normal cursor-pointer">
                    <input
                      type="radio"
                      name="viewMode"
                      checked={viewMode === "single"}
                      onChange={() => {
                        setViewMode("single");
                        if (!symbol) setSymbol("AAPL");
                      }}
                      className="mr-1"
                    />
                    Single Stock
                  </Label>
                  <Label className="text-sm font-normal cursor-pointer">
                    <input
                      type="radio"
                      name="viewMode"
                      checked={viewMode === "all"}
                      onChange={() => setViewMode("all")}
                      className="mr-1"
                    />
                    All Stocks
                  </Label>
                </div>
              </div>
              {viewMode === "single" ? (
                <StockSelector
                  selectedStock={symbol || "AAPL"}
                  onSelectStock={handleSelectStock}
                  searchQuery={searchQuery}
                  onSearchChange={setSearchQuery}
                />
              ) : (
                <div className="p-3 border rounded-md bg-muted/50">
                  <div className="text-sm font-medium">Showing predictions for all stocks</div>
                  {uniqueSymbols.length > 0 && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Found {uniqueSymbols.length} stock{uniqueSymbols.length !== 1 ? 's' : ''}: {uniqueSymbols.slice(0, 5).join(', ')}{uniqueSymbols.length > 5 ? '...' : ''}
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="interval">Interval</Label>
              <Select value={interval} onValueChange={setInterval}>
                <SelectTrigger id="interval">
                  <SelectValue placeholder="All intervals" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All intervals</SelectItem>
                  <SelectItem value="5m">5m</SelectItem>
                  <SelectItem value="15m">15m</SelectItem>
                  <SelectItem value="1h">1h</SelectItem>
                  <SelectItem value="4h">4h</SelectItem>
                  <SelectItem value="1d">1d</SelectItem>
                  <SelectItem value="1wk">1wk</SelectItem>
                  <SelectItem value="1mo">1mo</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="limit">Limit</Label>
              <Input
                id="limit"
                type="number"
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value) || 100)}
                min={1}
                max={1000}
              />
            </div>
            <div className="space-y-2">
              <Label>&nbsp;</Label>
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  {viewMode === "single" && symbol && (
                    <Button onClick={handleEvaluate} disabled={evaluating || evaluatingAll}>
                      {evaluating ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4 mr-2" />
                      )}
                      Evaluate {symbol}
                    </Button>
                  )}
                  <Button variant="outline" onClick={handleRefresh} disabled={historyLoading}>
                    {historyLoading ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-2" />
                    )}
                    Refresh
                  </Button>
                </div>
                <Button 
                  onClick={handleEvaluateAll} 
                  disabled={evaluating || evaluatingAll}
                  variant="secondary"
                  className="w-full"
                >
                  {evaluatingAll ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Zap className="h-4 w-4 mr-2" />
                  )}
                  Evaluate All Stocks
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Direction Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            ) : (
              <div className="text-3xl font-bold">
                {stats?.direction_accuracy?.toFixed(1) || 0}%
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              {stats?.total || 0} predictions evaluated
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Average Error</CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            ) : (
              <div className="text-3xl font-bold">
                {stats?.avg_error_percent?.toFixed(2) || 0}%
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              RMSE: ${stats?.rmse?.toFixed(2) || 0}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Mean Absolute Error</CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            ) : (
              <div className="text-3xl font-bold">
                ${stats?.mae?.toFixed(2) || 0}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Average absolute price difference
            </p>
          </CardContent>
        </Card>
      </div>
      
      {/* Prediction History Table */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction History</CardTitle>
          <CardDescription>
            Historical predictions with evaluation results
          </CardDescription>
        </CardHeader>
        <CardContent>
          {historyLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : !history || history.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <p>No predictions found {viewMode === "all" ? "for any stocks" : `for ${symbol}`}</p>
              <p className="text-sm mt-2">Make some predictions first to see them here</p>
            </div>
          ) : viewMode === "all" && uniqueSymbols.length === 1 ? (
            <div className="text-center py-12">
              <div className="text-muted-foreground mb-4">
                <p className="text-lg font-medium">Only predictions for {uniqueSymbols[0]} found</p>
                <p className="text-sm mt-2">To see multiple stocks:</p>
                <ol className="text-sm mt-2 text-left inline-block">
                  <li>1. Go to Dashboard and make predictions for other stocks (MSFT, GOOGL, etc.)</li>
                  <li>2. Return here and switch to "All Stocks" view</li>
                  <li>3. You'll see predictions from all stocks</li>
                </ol>
              </div>
              <div className="overflow-x-auto mt-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Interval</TableHead>
                      <TableHead>Current Price</TableHead>
                      <TableHead>Predicted Price</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Predicted Direction</TableHead>
                      <TableHead>Actual Price</TableHead>
                      <TableHead>Actual Direction</TableHead>
                      <TableHead>Error</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="w-[100px]">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {history.map((item: PredictionHistoryItem) => (
                      <TableRow key={item.id}>
                        <TableCell className="font-mono text-sm">
                          {formatDate(item.timestamp)}
                        </TableCell>
                        <TableCell className="font-semibold">{item.symbol}</TableCell>
                        <TableCell>{item.interval}</TableCell>
                        <TableCell>{formatPrice(item.current_price)}</TableCell>
                        <TableCell className="font-semibold">
                          {formatPrice(item.predicted_price)}
                        </TableCell>
                        <TableCell>
                          <Badge variant="secondary">
                            {item.confidence.toFixed(1)}%
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getDirectionIcon(item.direction_predicted)}
                            {getDirectionBadge(item.direction_predicted)}
                          </div>
                        </TableCell>
                        <TableCell>
                          {item.actual_price ? (
                            formatPrice(item.actual_price)
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {item.direction_actual ? (
                            <div className="flex items-center gap-2">
                              {getDirectionIcon(item.direction_actual)}
                              {getDirectionBadge(item.direction_actual)}
                            </div>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {item.error !== undefined ? (
                            <div className="flex flex-col">
                              <span className={item.error_percent && item.error_percent > 5 ? "text-red-500" : "text-green-500"}>
                                {formatPrice(item.error)}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                ({item.error_percent?.toFixed(2)}%)
                              </span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {item.evaluated ? (
                            item.correct ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-red-500" />
                            )
                          ) : (
                            <Badge variant="outline">Pending</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(item.id)}
                            className="text-destructive hover:text-destructive"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Interval</TableHead>
                    <TableHead>Current Price</TableHead>
                    <TableHead>Predicted Price</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Predicted Direction</TableHead>
                    <TableHead>Actual Price</TableHead>
                    <TableHead>Actual Direction</TableHead>
                    <TableHead>Error</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="w-[100px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.map((item: PredictionHistoryItem) => (
                    <TableRow key={item.id}>
                      <TableCell className="font-mono text-sm">
                        {formatDate(item.timestamp)}
                      </TableCell>
                      <TableCell className="font-semibold">{item.symbol}</TableCell>
                      <TableCell>{item.interval}</TableCell>
                      <TableCell>{formatPrice(item.current_price)}</TableCell>
                      <TableCell className="font-semibold">
                        {formatPrice(item.predicted_price)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {item.confidence.toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getDirectionIcon(item.direction_predicted)}
                          {getDirectionBadge(item.direction_predicted)}
                        </div>
                      </TableCell>
                      <TableCell>
                        {item.actual_price ? (
                          formatPrice(item.actual_price)
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {item.direction_actual ? (
                          <div className="flex items-center gap-2">
                            {getDirectionIcon(item.direction_actual)}
                            {getDirectionBadge(item.direction_actual)}
                          </div>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {item.error !== undefined ? (
                          <div className="flex flex-col">
                            <span className={item.error_percent && item.error_percent > 5 ? "text-red-500" : "text-green-500"}>
                              {formatPrice(item.error)}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              ({item.error_percent?.toFixed(2)}%)
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {item.evaluated ? (
                          item.correct ? (
                            <CheckCircle2 className="h-5 w-5 text-green-500" />
                          ) : (
                            <XCircle className="h-5 w-5 text-red-500" />
                          )
                        ) : (
                          <Badge variant="outline">Pending</Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(item.id)}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Prediction?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this prediction? This action cannot be undone.
              The associated evaluation will also be deleted.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

