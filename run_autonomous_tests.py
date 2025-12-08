#!/usr/bin/env python3
"""
Autonomous Test Runner - Complete testing pipeline
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class AutonomousTestRunner:
    """Complete autonomous testing system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_url = "http://localhost:5001"
        self.frontend_url = "http://localhost:60000"
        self.test_results = {}
        self.services_started = False

    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colored_message = self._colorize_message(message, level)
        print(f"[{timestamp}] {level}: {colored_message}")

    def _colorize_message(self, message: str, level: str) -> str:
        """Add ANSI color codes based on log level"""
        colors = {
            "INFO": "\033[34m",    # Blue
            "SUCCESS": "\033[32m", # Green
            "WARNING": "\033[33m", # Yellow
            "ERROR": "\033[31m",   # Red
            "RESET": "\033[0m"     # Reset
        }

        color = colors.get(level, colors["RESET"])
        return f"{color}{message}{colors['RESET']}"

    def ensure_services_running(self) -> bool:
        """Ensure all required services are running"""
        self.log("Checking service health...")

        # Check backend
        try:
            response = requests.get(f"{self.backend_url}/api/health", timeout=5)
            if response.status_code == 200:
                self.log("✅ Backend service is healthy", "SUCCESS")
            else:
                self.log(f"⚠️ Backend returned status {response.status_code}", "WARNING")
                return False
        except requests.exceptions.RequestException:
            self.log("❌ Backend service not responding", "ERROR")
            return False

        # Check frontend
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                self.log("✅ Frontend service is healthy", "SUCCESS")
            else:
                self.log(f"⚠️ Frontend returned status {response.status_code}", "WARNING")
        except requests.exceptions.RequestException:
            self.log("⚠️ Frontend service not responding (may be starting)", "WARNING")

        return True

    def start_services(self):
        """Start required services if not running"""
        if self.ensure_services_running():
            self.log("All services are already running")
            return

        self.log("Starting services...")

        # Start backend
        self.log("Starting backend server...")
        backend_process = subprocess.Popen(
            [sys.executable, "scripts/start_api.py"],
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for backend to start
        self.log("Waiting for backend to initialize...")
        time.sleep(5)

        # Check if backend started successfully
        if self.ensure_services_running():
            self.services_started = True
            self.log("✅ Services started successfully", "SUCCESS")
        else:
            self.log("❌ Failed to start services", "ERROR")
            # Kill the process if it didn't start properly
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
            sys.exit(1)

    def generate_test_data(self) -> bool:
        """Generate fresh test data"""
        self.log("Generating test data...")

        try:
            response = requests.post(
                f"{self.backend_url}/api/generate-data",
                json={"n_steps": 200, "description": "autonomous_test_data"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Test data generated: {data.get('shape', {})}", "SUCCESS")
                return True
            else:
                self.log(f"❌ Data generation failed: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Data generation error: {e}", "ERROR")
            return False

    def run_model_tests(self) -> Dict[str, Any]:
        """Run comprehensive model testing"""
        self.log("Running model tests...")

        test_results = {
            "neural": {},
            "classical": {},
            "advanced": {}
        }

        # Test configurations
        test_configs = {
            "neural": [
                ("/api/predict/forecasting", "Load Forecasting"),
                ("/api/anomaly-detection", "Anomaly Detection"),
            ],
            "classical": [
                ("/api/classical/load-forecasting", "Load Forecasting"),
                ("/api/classical/state-estimation", "State Estimation"),
                ("/api/classical/congestion-prediction", "Congestion Prediction"),
                ("/api/classical/opf-solver", "OPF Solver"),
                ("/api/classical/spatiotemporal", "Spatiotemporal"),
                ("/api/classical/anomaly-detection", "Anomaly Detection"),
            ],
            "advanced": [
                ("/api/advanced/forecasting", "Bayesian Structural Time Series"),
                ("/api/advanced/state_estimation", "Extended Kalman Filter"),
                ("/api/advanced/congestion", "MCMC Logistic Regression"),
                ("/api/advanced/opf", "Interior Point OPF"),
                ("/api/advanced/spatiotemporal", "Gaussian Process PDE"),
                ("/api/advanced/anomaly", "Hidden Markov Change Point"),
            ]
        }

        # Run tests for each category
        for category, endpoints in test_configs.items():
            self.log(f"Testing {category} models...")

            for endpoint, name in endpoints:
                self.log(f"  Testing {name}...")

                try:
                    if "anomaly-detection" in endpoint or "anomaly" in endpoint:
                        # GET request for anomaly detection
                        response = requests.get(f"{self.backend_url}{endpoint}", timeout=15)
                    else:
                        # POST request for other endpoints
                        response = requests.post(f"{self.backend_url}{endpoint}", timeout=15)

                    success = response.status_code == 200
                    if success:
                        try:
                            data = response.json()
                            success = data.get('success', False)
                        except:
                            success = False

                    test_results[category][name] = {
                        "success": success,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }

                    status_icon = "✅" if success else "❌"
                    self.log(f"    {status_icon} {name}: {'PASS' if success else 'FAIL'}")

                except Exception as e:
                    test_results[category][name] = {
                        "success": False,
                        "error": str(e)
                    }
                    self.log(f"    ❌ {name}: ERROR - {e}", "ERROR")

        return test_results

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run basic performance tests"""
        self.log("Running performance tests...")

        performance_results = {}

        # Test response times for key endpoints
        endpoints_to_test = [
            ("/api/health", "Health Check"),
            ("/api/advanced/forecasting", "Advanced Forecasting"),
            ("/api/classical/congestion-prediction", "Classical Congestion"),
        ]

        for endpoint, name in endpoints_to_test:
            response_times = []

            # Run 5 requests to get average
            for i in range(5):
                try:
                    start_time = time.time()
                    if "health" in endpoint:
                        response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                    else:
                        response = requests.post(f"{self.backend_url}{endpoint}", timeout=15)
                    end_time = time.time()

                    if response.status_code == 200:
                        response_times.append(end_time - start_time)

                except Exception as e:
                    self.log(f"Performance test error for {name}: {e}", "WARNING")

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                performance_results[name] = {
                    "avg_response_time": avg_time,
                    "max_response_time": max_time,
                    "success_rate": len(response_times) / 5
                }
                self.log(".2f")
            else:
                performance_results[name] = {"error": "No successful requests"}

        return performance_results

    def generate_report(self, model_results: Dict, perf_results: Dict) -> str:
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
{'='*80}
AUTONOMOUS TEST REPORT - {timestamp}
{'='*80}

SERVICES STATUS:
{'✅' if self.ensure_services_running() else '❌'} Backend: {self.backend_url}
{'✅' if self._check_frontend() else '❌'} Frontend: {self.frontend_url}

MODEL TEST RESULTS:
"""

        # Count results
        neural_passed = sum(1 for r in model_results['neural'].values() if r['success'])
        neural_total = len(model_results['neural'])

        classical_passed = sum(1 for r in model_results['classical'].values() if r['success'])
        classical_total = len(model_results['classical'])

        advanced_passed = sum(1 for r in model_results['advanced'].values() if r['success'])
        advanced_total = len(model_results['advanced'])

        total_passed = neural_passed + classical_passed + advanced_passed
        total_tests = neural_total + classical_total + advanced_total

        report += f"""
Neural Models: {neural_passed}/{neural_total} passed
Classical Models: {classical_passed}/{classical_total} passed
Advanced Models: {advanced_passed}/{advanced_total} passed

OVERALL RESULT: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}% success rate)

DETAILED RESULTS:
"""

        for category, results in model_results.items():
            report += f"\n{category.upper()} MODELS:\n"
            for name, result in results.items():
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                response_time = f" ({result.get('response_time', 0):.2f}s)" if 'response_time' in result else ""
                report += f"  {status} {name}{response_time}\n"

        report += f"\nPERFORMANCE RESULTS:\n"
        for name, perf in perf_results.items():
            if 'avg_response_time' in perf:
                report += ".2f"
            else:
                report += f"  ❌ {name}: {perf.get('error', 'Unknown error')}\n"

        report += f"\n{'='*80}\n"

        # Save report to file
        report_file = self.project_root / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        self.log(f"📄 Report saved to: {report_file}", "SUCCESS")

        return report

    def _check_frontend(self) -> bool:
        """Check if frontend is responding"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def run_full_test_suite(self) -> bool:
        """Run complete autonomous test suite"""
        self.log("🚀 Starting Autonomous Test Suite")
        self.log("=" * 60)

        try:
            # Start services
            self.start_services()

            # Generate test data
            if not self.generate_test_data():
                self.log("❌ Test data generation failed", "ERROR")
                return False

            # Run model tests
            model_results = self.run_model_tests()

            # Run performance tests
            perf_results = self.run_performance_tests()

            # Generate report
            report = self.generate_report(model_results, perf_results)

            # Print summary
            print(report)

            # Calculate success rate
            all_results = []
            for category in ['neural', 'classical', 'advanced']:
                all_results.extend(model_results[category].values())

            success_rate = sum(1 for r in all_results if r['success']) / len(all_results)

            if success_rate >= 0.8:  # 80% success threshold
                self.log("🎉 Test suite completed successfully!", "SUCCESS")
                return True
            else:
                self.log(f"⚠️ Test suite completed with low success rate: {success_rate:.1%}", "WARNING")
                return False

        except Exception as e:
            self.log(f"💥 Test suite crashed: {e}", "ERROR")
            return False

    def cleanup(self):
        """Clean up resources"""
        self.log("Cleaning up...")
        # Services will be cleaned up by the calling process

def main():
    """Main entry point"""
    runner = AutonomousTestRunner()

    try:
        success = runner.run_full_test_suite()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️ Test suite interrupted by user")
        runner.cleanup()
        sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        runner.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
