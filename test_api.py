#!/usr/bin/env python3
"""
Quick API test script
"""

import requests
import time

def test_api():
    print("Testing API endpoints...")

    base_url = "http://localhost:5001"

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")

    # Test data generation
    try:
        response = requests.post(f"{base_url}/api/generate-data", json={"n_steps": 100})
        if response.status_code == 200:
            print("Data generation working")
        else:
            print(f"❌ Data generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data generation error: {e}")

    # Test classical model
    try:
        response = requests.post(f"{base_url}/api/classical/load-forecasting")
        if response.status_code == 200:
            print("Classical model working")
        else:
            print(f"❌ Classical model failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classical model error: {e}")

if __name__ == "__main__":
    test_api()
