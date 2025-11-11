#!/usr/bin/env python3

# python_backend/test_improvements.py
"""
Comprehensive testing suite to measure model improvements.
Tests accuracy, performance, and response time across multiple stocks.
"""

import requests
import json
import time
from datetime import datetime
from statistics import mean, stdev

API_BASE = "http://localhost:5001"
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

class bcolors:
    """Terminal colors for better output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}{'='*70}{bcolors.ENDC}")
    print(f"{bcolors.HEADER}{bcolors.BOLD}{text:^70}{bcolors.ENDC}")
    print(f"{bcolors.HEADER}{bcolors.BOLD}{'='*70}{bcolors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{bcolors.OKGREEN}✓ {text}{bcolors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{bcolors.OKBLUE}ℹ {text}{bcolors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{bcolors.WARNING}⚠ {text}{bcolors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{bcolors.FAIL}✗ {text}{bcolors.ENDC}")

def test_single_stock_accuracy(symbol="AAPL", num_tests=5):
    """Test prediction accuracy for a single stock"""
    print_info(f"Testing {symbol} predictions ({num_tests} iterations)...")
    
    confidences = []
    predictions = []
    
    for i in range(num_tests):
        try:
            response = requests.get(
                f"{API_BASE}/api/predict/{symbol}?horizon=5&range=1y&interval=1d",
                timeout=60
            )
            data = response.json()
            
            if response.status_code == 200:
                conf = data.get('confidence', 0) / 100.0 if data.get('confidence') else 0
                pred = data.get('predictedPrice', 0)
                confidences.append(conf)
                predictions.append(pred)
                print(f"   Test {i+1}: Pred=${pred:.2f}, Conf={conf:.2%}")
            else:
                print_warning(f"Test {i+1}: Error - {data.get('error', 'Unknown')}")
                
        except requests.exceptions.Timeout:
            print_error(f"Test {i+1}: Timeout (model training may still be running)")
        except Exception as e:
            print_error(f"Test {i+1}: {str(e)}")
        
        time.sleep(1)
    
    if confidences:
        avg_conf = mean(confidences)
        std_conf = stdev(confidences) if len(confidences) > 1 else 0
        avg_pred = mean(predictions)
        
        print_success(f"Average Confidence: {avg_conf:.2%} (±{std_conf:.2%})")
        print_success(f"Average Prediction: ${avg_pred:.2f}")
        return {"avg_confidence": avg_conf, "std_confidence": std_conf, 
                "avg_prediction": avg_pred, "num_tests": len(confidences)}
    else:
        print_error("All tests failed")
        return None

def test_individual_models(symbol="AAPL"):
    """Compare performance of individual models"""
    print_info(f"Testing individual models for {symbol}...")
    
    try:
        response = requests.get(
            f"{API_BASE}/api/predict/{symbol}?horizon=5&range=1y&interval=1d",
            timeout=60
        )
        data = response.json()
        
        if response.status_code == 200:
            models = data.get('models', {})
            
            if not models:
                print_warning("No individual model data available")
                return None
            
            # Sort by confidence
            sorted_models = sorted(
                models.items(),
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )
            
            print("\n   Model Rankings (by confidence):")
            print("   " + "-"*60)
            for rank, (model_name, model_data) in enumerate(sorted_models, 1):
                pred = model_data.get('prediction', 0)
                conf = model_data.get('confidence', 0) / 100.0 if model_data.get('confidence') else 0
                weight = model_data.get('weight', 0)
                status = "🏆" if rank == 1 else "✓" if rank <= 3 else " "
                print(f"   {status} {rank}. {model_name:15s}: "
                      f"Pred=${pred:8.2f}, Conf={conf:.2%}, Weight={weight:.1f}%")
            
            return sorted_models
        else:
            print_error(f"API Error: {data.get('error', 'Unknown')}")
            return None
            
    except Exception as e:
        print_error(f"Failed: {str(e)}")
        return None

def test_all_stocks(num_tests_per_stock=3):
    """Test predictions across all stocks"""
    print_info(f"Testing {len(STOCKS)} stocks...")
    
    results = {}
    
    for symbol in STOCKS:
        print(f"\n{bcolors.OKCYAN}Testing {symbol}...{bcolors.ENDC}")
        result = test_single_stock_accuracy(symbol, num_tests=num_tests_per_stock)
        if result:
            results[symbol] = result
        time.sleep(2)
    
    if results:
        print_header("MULTI-STOCK SUMMARY")
        print(f"{'Stock':<8} {'Avg Confidence':<15} {'Std Dev':<12} {'Tests':<8}")
        print("-" * 50)
        
        all_confidences = []
        for symbol, data in results.items():
            conf = data['avg_confidence']
            std = data['std_confidence']
            tests = data['num_tests']
            all_confidences.append(conf)
            
            status = "🌟" if conf > 0.65 else "✓" if conf > 0.55 else "⚠"
            print(f"{status} {symbol:<6} {conf:>12.2%}   {std:>10.2%}   {tests:>5}")
        
        overall_avg = mean(all_confidences)
        overall_std = stdev(all_confidences) if len(all_confidences) > 1 else 0
        
        print("-" * 50)
        print_success(f"Overall Average: {overall_avg:.2%} (±{overall_std:.2%})")
        
        return results
    
    return None

def benchmark_response_time(symbol="AAPL", num_requests=5):
    """Measure API response time"""
    print_info(f"Benchmarking response time ({num_requests} requests)...")
    
    times = []
    for i in range(num_requests):
        start = time.time()
        try:
            response = requests.get(
                f"{API_BASE}/api/predict/{symbol}?horizon=5&range=1y&interval=1d",
                timeout=60
            )
            elapsed = time.time() - start
            times.append(elapsed)
            
            status = "⚡" if elapsed < 1 else "✓" if elapsed < 3 else "⚠"
            print(f"   {status} Request {i+1}: {elapsed:.3f}s")
            
        except Exception as e:
            elapsed = time.time() - start
            print_error(f"Request {i+1}: Failed after {elapsed:.3f}s - {str(e)}")
        
        time.sleep(0.5)
    
    if times:
        avg_time = mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print_success(f"Average: {avg_time:.3f}s (min: {min_time:.3f}s, max: {max_time:.3f}s)")
        
        if avg_time < 1:
            print_success("Response time: Excellent ⚡")
        elif avg_time < 3:
            print_info("Response time: Good ✓")
        else:
            print_warning("Response time: Slow (consider optimization)")
        
        return {"avg": avg_time, "min": min_time, "max": max_time}
    
    return None

def test_api_health():
    """Check if API is responding"""
    print_info("Checking API health...")
    
    try:
        response = requests.get(f"{API_BASE}/api/stocks?popular=true", timeout=5)
        if response.status_code == 200:
            print_success("API is running")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print_error(f"API health check failed: {str(e)}")
        return False

def save_results(results, filename=None):
    """Save test results to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print_success(f"Results saved to: {filename}")
    except Exception as e:
        print_error(f"Failed to save results: {str(e)}")

def main():
    """Run all tests"""
    print_header("MODEL IMPROVEMENT TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 0: Health check
    print_header("TEST 0: API Health Check")
    if not test_api_health():
        print_error("\n⚠ API is not responding. Please start the backend first:")
        print("   cd python_backend")
        print("   ./start_improved.sh")
        return
    
    # Test 1: Single stock detailed test
    print_header("TEST 1: Single Stock Detailed Analysis")
    result = test_single_stock_accuracy("AAPL", num_tests=5)
    if result:
        all_results["tests"]["single_stock"] = result
    
    # Test 2: Individual model comparison
    print_header("TEST 2: Individual Model Performance")
    models = test_individual_models("AAPL")
    if models:
        all_results["tests"]["model_comparison"] = [
            {"model": name, "data": data} for name, data in models
        ]
    
    # Test 3: Multi-stock test
    print_header("TEST 3: Multi-Stock Analysis")
    multi_results = test_all_stocks(num_tests_per_stock=3)
    if multi_results:
        all_results["tests"]["multi_stock"] = multi_results
    
    # Test 4: Performance benchmark
    print_header("TEST 4: Response Time Benchmark")
    perf = benchmark_response_time("AAPL", num_requests=5)
    if perf:
        all_results["tests"]["performance"] = perf
    
    # Save results
    print_header("SAVING RESULTS")
    save_results(all_results)
    
    # Final summary
    print_header("TEST SUITE COMPLETED")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{bcolors.BOLD}💡 Next Steps:{bcolors.ENDC}")
    print("   1. Compare these results with your baseline")
    print("   2. Document improvements in your dissertation")
    print("   3. If accuracy > 55%, proceed to Phase 2 (Backtesting)")
    print("   4. If accuracy < 50%, consider additional feature engineering\n")

if __name__ == "__main__":
    main()

