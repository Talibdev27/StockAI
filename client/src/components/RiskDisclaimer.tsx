import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function RiskDisclaimer() {
  return (
    <Alert className="border-yellow-500/50 bg-yellow-500/10">
      <AlertTriangle className="h-4 w-4 text-yellow-500" />
      <AlertDescription className="text-sm">
        <span className="font-semibold">Risk Disclaimer:</span> This platform is for educational purposes only and does not constitute financial advice. 
        Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
      </AlertDescription>
    </Alert>
  );
}
