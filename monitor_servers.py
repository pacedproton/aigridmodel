#!/usr/bin/env python3
"""
Server monitoring and health check script
Continuously monitors backend and frontend servers
"""

import requests
import time
import subprocess
import signal
import sys
from datetime import datetime

class ServerMonitor:
    def __init__(self):
        self.backend_url = "http://localhost:5001"
        self.frontend_url = "http://localhost:60000"
        self.running = True
        self.backend_process = None
        self.frontend_process = None

    def check_service(self, url: str, name: str) -> str:
        """Check if a service is responding"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return f"✅ {name}: UP"
            else:
                return f"⚠️  {name}: HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"❌ {name}: DOWN ({str(e)[:30]}...)"

    def check_api_endpoints(self) -> None:
        """Test key API endpoints"""
        endpoints = [
            ("/api/health", "Health"),
            ("/api/generate-data", "Data Gen", "POST", {"n_steps": 50}),
            ("/api/advanced/forecasting", "Adv Forecasting", "POST"),
        ]

        print(f"\n🔍 API Endpoint Tests ({datetime.now().strftime('%H:%M:%S')}):")
        for endpoint, name, method, *data in endpoints:
            try:
                if method == "POST":
                    json_data = data[0] if data else None
                    response = requests.post(f"{self.backend_url}{endpoint}",
                                           json=json_data, timeout=10)
                else:
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)

                status = "✅" if response.status_code == 200 else f"❌{response.status_code}"
                print(f"  {status} {name}")
            except Exception as e:
                print(f"  ❌ {name}: {str(e)[:25]}...")

    def start_servers(self):
        """Start backend and frontend servers"""
        print("🚀 Starting AI Grid Model Servers...")

        try:
            # Start backend
            print("📡 Starting backend server...")
            self.backend_process = subprocess.Popen(
                ["python", "scripts/start_api.py"],
                cwd=".",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for backend to start
            time.sleep(3)

            # Start frontend
            print("🌐 Starting frontend server...")
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd="frontend",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for frontend to start
            time.sleep(5)

        except Exception as e:
            print(f"❌ Failed to start servers: {e}")
            self.stop_servers()
            sys.exit(1)

    def stop_servers(self):
        """Stop running servers"""
        print("\n🛑 Stopping servers...")

        if self.backend_process and self.backend_process.poll() is None:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                print("⚠️ Backend force killed")

        if self.frontend_process and self.frontend_process.poll() is None:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                print("⚠️ Frontend force killed")

    def monitor_loop(self):
        """Main monitoring loop"""
        print("📊 Server Monitoring Active")
        print("Press Ctrl+C to stop monitoring and servers")
        print("=" * 50)

        cycle = 0
        while self.running:
            try:
                # Service health checks
                backend_status = self.check_service(f"{self.backend_url}/api/health", "Backend")
                frontend_status = self.check_service(self.frontend_url, "Frontend")

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Server Status:")
                print(f"  {backend_status}")
                print(f"  {frontend_status}")

                # Detailed API checks every 30 seconds
                if cycle % 6 == 0:  # Every 30 seconds (5 second intervals)
                    self.check_api_endpoints()

                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("⚠️ Backend process terminated unexpectedly!"                    break

                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("⚠️ Frontend process terminated unexpectedly!"                    break

                cycle += 1
                time.sleep(5)

            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(2)

    def run(self):
        """Main run method"""
        signal.signal(signal.SIGINT, lambda sig, frame: self.handle_interrupt())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.handle_interrupt())

        try:
            self.start_servers()
            self.monitor_loop()
        finally:
            self.stop_servers()

    def handle_interrupt(self):
        """Handle interrupt signals"""
        self.running = False

def main():
    """Main entry point"""
    monitor = ServerMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
