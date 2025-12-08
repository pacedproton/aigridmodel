#!/usr/bin/env python3
"""
Comprehensive test suite for AI Grid Model API
Tests all endpoints systematically and reports results
"""

import requests
import time
import json
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {
            'health': False,
            'data_generation': False,
            'neural_models': {},
            'classical_models': {},
            'advanced_models': {},
            'errors': []
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, description: str = "") -> Dict[str, Any]:
        """Test a single endpoint and return result"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=30)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}

            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "description": description
            }

            if response.status_code == 200:
                try:
                    result["data"] = response.json()
                except:
                    result["data"] = {"raw_response": response.text[:200]}
            else:
                result["error"] = response.text
                self.results['errors'].append(f"{endpoint}: {response.text}")

            return result

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            self.results['errors'].append(f"{endpoint}: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "description": description
            }

    def run_all_tests(self):
        """Run comprehensive test suite"""
        self.log("🚀 Starting comprehensive API test suite")
        self.log("=" * 50)

        # 1. Test health endpoint
        self.log("Testing health endpoint...")
        health_result = self.test_endpoint("GET", "/api/health", description="API health check")
        self.results['health'] = health_result['success']
        self.log(f"Health check: {'✅ PASS' if health_result['success'] else '❌ FAIL'}")

        if not health_result['success']:
            self.log("❌ Backend server not responding. Stopping tests.", "ERROR")
            return self.results

        # 2. Generate test data
        self.log("Generating test data...")
        data_result = self.test_endpoint("POST", "/api/generate-data",
                                       {"n_steps": 200}, "Data generation")
        self.results['data_generation'] = data_result['success']
        self.log(f"Data generation: {'✅ PASS' if data_result['success'] else '❌ FAIL'}")

        if not data_result['success']:
            self.log("⚠️ Data generation failed, but continuing with tests", "WARN")

        # 3. Test neural network models
        self.log("Testing Neural Network models...")
        neural_endpoints = [
            ("/api/predict/forecasting", "Load Forecasting"),
            ("/api/anomaly-detection", "Anomaly Detection"),
        ]

        for endpoint, description in neural_endpoints:
            self.log(f"  Testing {description}...")
            result = self.test_endpoint("POST", endpoint, description=f"Neural {description}")
            self.results['neural_models'][description] = result['success']
            self.log(f"    {description}: {'✅ PASS' if result['success'] else '❌ FAIL'}")

        # 4. Test classical models
        self.log("Testing Classical models...")
        classical_endpoints = [
            ("/api/classical/load-forecasting", "Load Forecasting"),
            ("/api/classical/state-estimation", "State Estimation"),
            ("/api/classical/congestion-prediction", "Congestion Prediction"),
            ("/api/classical/opf-solver", "OPF Solver"),
            ("/api/classical/spatiotemporal", "Spatiotemporal"),
            ("/api/classical/anomaly-detection", "Anomaly Detection"),
        ]

        for endpoint, description in classical_endpoints:
            self.log(f"  Testing {description}...")
            result = self.test_endpoint("POST", endpoint, description=f"Classical {description}")
            self.results['classical_models'][description] = result['success']
            self.log(f"    {description}: {'✅ PASS' if result['success'] else '❌ FAIL'}")

        # 5. Test advanced mathematical models
        self.log("Testing Advanced Mathematical models...")
        advanced_endpoints = [
            ("forecasting", "Bayesian Structural Time Series"),
            ("state_estimation", "Extended Kalman Filter"),
            ("congestion", "MCMC Logistic Regression"),
            ("opf", "Interior Point OPF"),
            ("spatiotemporal", "Gaussian Process PDE"),
            ("anomaly", "Hidden Markov Change Point"),
        ]

        for model_type, description in advanced_endpoints:
            endpoint = f"/api/advanced/{model_type}"
            self.log(f"  Testing {description}...")
            result = self.test_endpoint("POST", endpoint, description=f"Advanced {description}")
            self.results['advanced_models'][description] = result['success']
            self.log(f"    {description}: {'✅ PASS' if result['success'] else '❌ FAIL'}")

        # Summary
        self.log("=" * 50)
        self.log("📊 TEST SUMMARY")
        self.log("=" * 50)

        self.log(f"Health Check: {'✅' if self.results['health'] else '❌'}")
        self.log(f"Data Generation: {'✅' if self.results['data_generation'] else '❌'}")

        self.log(f"\nNeural Models ({sum(self.results['neural_models'].values())}/{len(self.results['neural_models'])}):")
        for model, success in self.results['neural_models'].items():
            self.log(f"  {model}: {'✅' if success else '❌'}")

        self.log(f"\nClassical Models ({sum(self.results['classical_models'].values())}/{len(self.results['classical_models'])}):")
        for model, success in self.results['classical_models'].items():
            self.log(f"  {model}: {'✅' if success else '❌'}")

        self.log(f"\nAdvanced Models ({sum(self.results['advanced_models'].values())}/{len(self.results['advanced_models'])}):")
        for model, success in self.results['advanced_models'].items():
            self.log(f"  {model}: {'✅' if success else '❌'}")

        total_passed = sum([
            self.results['health'],
            self.results['data_generation'],
            sum(self.results['neural_models'].values()),
            sum(self.results['classical_models'].values()),
            sum(self.results['advanced_models'].values())
        ])

        total_tests = 2 + len(self.results['neural_models']) + len(self.results['classical_models']) + len(self.results['advanced_models'])

        self.log(f"\n🎯 OVERALL RESULT: {total_passed}/{total_tests} tests passed")

        if self.results['errors']:
            self.log(f"\n❌ ERRORS ENCOUNTERED ({len(self.results['errors'])}):")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                self.log(f"  • {error}")
            if len(self.results['errors']) > 5:
                self.log(f"  ... and {len(self.results['errors']) - 5} more errors")

        return self.results

def main():
    """Main test runner"""
    tester = APITester()

    try:
        results = tester.run_all_tests()

        # Exit with appropriate code
        if results['health'] and sum(results['advanced_models'].values()) > 0:
            print("\n🎉 Core functionality working!")
            sys.exit(0)
        else:
            print("\n❌ Critical failures detected!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
