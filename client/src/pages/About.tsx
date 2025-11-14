import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Brain, 
  TrendingUp, 
  TestTube, 
  Zap, 
  Shield, 
  Target,
  LineChart,
  Database,
  Cpu,
  CheckCircle2,
  History,
  Activity,
  Gauge,
  Trash2
} from "lucide-react";

export default function About() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12 max-w-6xl space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            StockPredict AI
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered stock market analysis and predictions using advanced machine learning models
          </p>
        </div>

        {/* Mission Statement */}
        <Card className="p-6 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
          <p className="text-lg text-center">
            We provide intelligent stock market predictions by combining multiple AI models to deliver 
            accurate forecasts, pattern recognition, and comprehensive backtesting capabilities for informed trading decisions.
          </p>
        </Card>

        {/* Features Grid */}
        <div>
          <h2 className="text-2xl font-semibold mb-6 text-center">Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Brain className="h-10 w-10 text-blue-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">7-Model Ensemble</h3>
              <p className="text-muted-foreground">
                Combines Linear Regression, LSTM neural networks, ARIMA time series, XGBoost, 
                Decision Tree, SVM, and Prophet models with performance-based weighting for 
                optimal prediction accuracy.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <LineChart className="h-10 w-10 text-green-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Advanced Charting</h3>
              <p className="text-muted-foreground">
                Professional TradingView-style charts with candlestick patterns, technical indicators, 
                and real-time price tracking across multiple timeframes.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <TestTube className="h-10 w-10 text-purple-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Backtesting Engine</h3>
              <p className="text-muted-foreground">
                Test your trading strategies with walk-forward analysis, comprehensive performance metrics, 
                and detailed equity curve visualization.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Target className="h-10 w-10 text-orange-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Pattern Recognition</h3>
              <p className="text-muted-foreground">
                Automatic detection of candlestick patterns (Doji, Hammer, Engulfing, etc.) with 
                strength scoring and visual indicators on charts.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Database className="h-10 w-10 text-cyan-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">S&P 500 Coverage</h3>
              <p className="text-muted-foreground">
                Access to all S&P 500 stocks with intelligent search and filtering capabilities 
                for easy stock discovery and analysis.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Zap className="h-10 w-10 text-yellow-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Real-Time Data</h3>
              <p className="text-muted-foreground">
                Live market data from Yahoo Finance with automatic updates, price quotes, 
                and historical data across multiple timeframes.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Gauge className="h-10 w-10 text-indigo-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Performance-Based Weighting</h3>
              <p className="text-muted-foreground">
                Dynamic model weights adapt based on historical prediction accuracy, 
                automatically prioritizing the most reliable models for each stock and timeframe.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <CheckCircle2 className="h-10 w-10 text-green-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Prediction Evaluation</h3>
              <p className="text-muted-foreground">
                Automatic evaluation system tracks prediction accuracy by comparing forecasts 
                with actual prices after time horizons pass, providing real performance metrics.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <History className="h-10 w-10 text-teal-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Prediction History</h3>
              <p className="text-muted-foreground">
                Complete history of all predictions with evaluation results, performance tracking, 
                and management tools. View, filter, and delete predictions with ease.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <Activity className="h-10 w-10 text-red-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Performance Metrics</h3>
              <p className="text-muted-foreground">
                Real-time tracking of prediction accuracy, Sharpe ratio, win rate, and total return 
                calculated from evaluated predictions to measure model effectiveness.
              </p>
            </Card>
          </div>
        </div>

        {/* Technology Stack */}
        <div>
          <h2 className="text-2xl font-semibold mb-6 text-center">Technology Stack</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Cpu className="h-5 w-5 text-blue-500" />
                Backend & AI
              </h3>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Python</Badge>
                <Badge variant="secondary">Flask</Badge>
                <Badge variant="secondary">TensorFlow</Badge>
                <Badge variant="secondary">Keras</Badge>
                <Badge variant="secondary">XGBoost</Badge>
                <Badge variant="secondary">scikit-learn</Badge>
                <Badge variant="secondary">statsmodels</Badge>
                <Badge variant="secondary">ARIMA</Badge>
                <Badge variant="secondary">LSTM</Badge>
                <Badge variant="secondary">Prophet</Badge>
                <Badge variant="secondary">PostgreSQL</Badge>
                <Badge variant="secondary">SQLite</Badge>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-green-500" />
                Frontend
              </h3>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">React</Badge>
                <Badge variant="secondary">TypeScript</Badge>
                <Badge variant="secondary">Vite</Badge>
                <Badge variant="secondary">Tailwind CSS</Badge>
                <Badge variant="secondary">Lightweight Charts</Badge>
                <Badge variant="secondary">Radix UI</Badge>
                <Badge variant="secondary">React Query</Badge>
              </div>
            </Card>
          </div>
        </div>

        {/* Key Metrics */}
        <div>
          <h2 className="text-2xl font-semibold mb-6 text-center">Capabilities</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-blue-500 mb-2">7</div>
              <div className="text-sm text-muted-foreground">ML Models</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-green-500 mb-2">500+</div>
              <div className="text-sm text-muted-foreground">S&P 500 Stocks</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-purple-500 mb-2">7</div>
              <div className="text-sm text-muted-foreground">Timeframes</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-orange-500 mb-2">10+</div>
              <div className="text-sm text-muted-foreground">Candlestick Patterns</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-indigo-500 mb-2">∞</div>
              <div className="text-sm text-muted-foreground">Predictions Tracked</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-3xl font-bold text-teal-500 mb-2">100%</div>
              <div className="text-sm text-muted-foreground">Auto Evaluation</div>
            </Card>
          </div>
        </div>

        {/* Disclaimer */}
        <Card className="p-6 bg-yellow-500/10 border-yellow-500/20">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-yellow-500 mt-0.5" />
            <div>
              <h3 className="font-semibold mb-2">Important Disclaimer</h3>
              <p className="text-sm text-muted-foreground">
                StockPredict AI is for educational and research purposes only. All predictions and 
                analyses are based on historical data and machine learning models, which may not accurately 
                forecast future market movements. Past performance does not guarantee future results. 
                Always conduct your own research and consult with financial advisors before making any 
                investment decisions. Trading stocks involves risk, including the potential loss of capital.
              </p>
            </div>
          </div>
        </Card>

        {/* Footer Info */}
        <div className="text-center text-sm text-muted-foreground space-y-2">
          <p>Built with ❤️ using cutting-edge AI and modern web technologies</p>
          <p>© 2025 StockPredict AI. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}

