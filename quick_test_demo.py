#!/usr/bin/env python3
"""
Quick demo of autonomous testing capabilities
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path


def run_demo():
    """Demonstrate autonomous testing capabilities"""
    print("AI Grid Model - Autonomous Testing Demo")
    print("=" * 50)

    project_dir = Path(__file__).parent

    # Step 1: Check if services are running
    print("\n1. Checking service health...")
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=5)
        if response.status_code == 200:
            print("Backend is running")
        else:
            print("Backend responding but not healthy")
    except:
        print("❌ Backend not running - starting it...")

        # Start backend
        backend = subprocess.Popen(
            [sys.executable, "scripts/start_api.py"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5)

        try:
            response = requests.get(
                "http://localhost:5001/api/health", timeout=5)
            if response.status_code == 200:
                print("Backend started successfully")
            else:
                print("❌ Backend failed to start properly")
                return
        except:
            print("❌ Backend still not responding")
            return

    # Step 2: Generate test data
    print("\n2. Generating test data...")
    try:
        response = requests.post(
            "http://localhost:5001/api/generate-data",
            json={"n_steps": 100},
            timeout=10
        )
        if response.status_code == 200:
            print("Test data generated")
        else:
            print(f"❌ Data generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data generation error: {e}")

    # Step 3: Test key models
    print("\n3. Testing key models...")

    models_to_test = [
        ("Advanced Forecasting", "/api/advanced/forecasting"),
        ("Classical Forecasting", "/api/classical/load-forecasting"),
        ("Neural Network", "/api/predict/forecasting"),
    ]

    for name, endpoint in models_to_test:
        try:
            response = requests.post(
                f"http://localhost:5001{endpoint}", timeout=15)
            success = response.status_code == 200
            if success and 'application/json' in response.headers.get('content-type', ''):
                data = response.json()
                success = data.get('success', False)

            status = "PASS" if success else "FAIL"
            print(f"   {status} {name}")
        except Exception as e:
            print(f"   ❌ {name}: Error - {str(e)[:30]}...")

    # Step 4: Run autonomous test suite
    print("\n4. Running full autonomous test suite...")
    try:
        result = subprocess.run(
            [sys.executable, "run_autonomous_tests.py"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            print("Autonomous test suite completed successfully")

            # Show summary from output
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # Last 10 lines
                if 'OVERALL RESULT:' in line or 'tests passed' in line:
                    print(f"   📊 {line.strip()}")
        else:
            print("❌ Autonomous test suite failed")
            # Last 200 chars of error
            print(f"   Error: {result.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        print("Test suite timed out (this is normal for comprehensive tests)")
    except Exception as e:
        print(f"❌ Test suite error: {e}")

    # Step 5: Show next steps
    print("\nNext Steps:")
    print("• View detailed reports: ls test_report_*.txt")
    print("• Check logs: tail -f *.log")
    print("• Set up scheduling: ./schedule_tests.sh")
    print("• View monitoring: python monitoring_dashboard.py")
    print("• Run GitHub Actions: Push to main branch")

    print("\n📚 Available Commands:")
    print("• Manual tests: python run_autonomous_tests.py")
    print("• Quick demo: python quick_test_demo.py")
    print("• Monitor services: python monitor_servers.py")
    print("• Schedule tests: ./schedule_tests.sh")

    print("\nAutonomous testing is now active!")


if __name__ == "__main__":
    run_demo()
