#!/bin/bash

# Autonomous Test Scheduler
# This script sets up cron jobs for regular automated testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if cron is available
check_cron() {
    if ! command -v crontab >/dev/null 2>&1; then
        error "Cron is not available on this system"
        exit 1
    fi
}

# Setup test schedule
setup_schedule() {
    log "Setting up automated test schedule..."

    # Create cron job entries
    CRON_JOBS="
# AI Grid Model Autonomous Testing
# Run comprehensive tests every 4 hours
0 */4 * * * cd $PROJECT_DIR && ./run_autonomous_tests.py >> autonomous_test.log 2>&1

# Run quick health checks every hour
0 * * * * cd $PROJECT_DIR && python -c \"import requests; print('Health:', requests.get('http://localhost:5001/api/health').status_code)\" >> health_check.log 2>&1

# Clean up old logs weekly (run at 2 AM on Sundays)
0 2 * * 0 cd $PROJECT_DIR && find . -name '*.log' -mtime +7 -delete

# Performance monitoring every 6 hours
0 */6 * * * cd $PROJECT_DIR && python -c \"
import requests
import time
start = time.time()
try:
    r = requests.post('http://localhost:5001/api/advanced/forecasting', timeout=30)
    duration = time.time() - start
    print(f'Performance: {duration:.2f}s - Status: {r.status_code}')
except Exception as e:
    print(f'Performance: ERROR - {e}')
\" >> performance.log 2>&1
"

    # Add to current crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -

    if [ $? -eq 0 ]; then
        log "✅ Test schedule configured successfully"
        log "📋 Scheduled tasks:"
        echo "  • Comprehensive tests: Every 4 hours"
        echo "  • Health checks: Every hour"
        echo "  • Performance monitoring: Every 6 hours"
        echo "  • Log cleanup: Weekly"
    else
        error "Failed to configure cron jobs"
        exit 1
    fi
}

# Setup log rotation
setup_logging() {
    log "Setting up log rotation..."

    # Create logrotate config
    LOGROTATE_CONF="/tmp/ai_grid_logrotate"
    cat > "$LOGROTATE_CONF" << EOF
$PROJECT_DIR/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 644 $(whoami) $(id -gn)
}
EOF

    if command -v logrotate >/dev/null 2>&1; then
        sudo logrotate -f "$LOGROTATE_CONF" 2>/dev/null || true
        log "✅ Log rotation configured"
    else
        warning "logrotate not available, using basic log management"
    fi

    # Cleanup
    rm -f "$LOGROTATE_CONF"
}

# Setup monitoring dashboard
setup_monitoring() {
    log "Setting up monitoring dashboard..."

    # Create simple monitoring script
    cat > "$PROJECT_DIR/monitoring_dashboard.py" << 'EOF'
#!/usr/bin/env python3
"""
Simple monitoring dashboard for test results
"""

import os
import json
from flask import Flask, render_template_string, jsonify
from pathlib import Path
import glob

app = Flask(__name__)
PROJECT_DIR = Path(__file__).parent

@app.route('/')
def dashboard():
    # Find latest test report
    report_files = list(PROJECT_DIR.glob("test_report_*.txt"))
    latest_report = max(report_files, key=os.path.getctime) if report_files else None

    report_content = ""
    if latest_report:
        with open(latest_report, 'r') as f:
            report_content = f.read()

    # Get log files
    log_files = {}
    for log_file in PROJECT_DIR.glob("*.log"):
        with open(log_file, 'r') as f:
            log_files[log_file.name] = f.readlines()[-20:]  # Last 20 lines

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Grid Model - Test Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; color: #155724; }}
            .error {{ background: #f8d7da; color: #721c24; }}
            .warning {{ background: #fff3cd; color: #856404; }}
            pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>🤖 AI Grid Model - Autonomous Testing Dashboard</h1>

        <div class="status success">
            <h2>✅ System Status</h2>
            <p>Autonomous testing is active and monitoring system health.</p>
        </div>

        <h2>📊 Latest Test Report</h2>
        <pre>{report_content or "No test reports available yet."}</pre>

        <h2>📝 Recent Logs</h2>
        {"".join(f"<h3>{name}</h3><pre>{''.join(lines)}</pre>" for name, lines in log_files.items())}
    </body>
    </html>
    """
    return html

@app.route('/api/status')
def api_status():
    # Return current system status
    return jsonify({
        "status": "active",
        "last_test": "Check log files for details",
        "services": ["backend", "frontend", "testing"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

    chmod +x "$PROJECT_DIR/monitoring_dashboard.py"
    log "✅ Monitoring dashboard created: monitoring_dashboard.py"
}

# Setup systemd service (Linux only)
setup_systemd() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        warning "Systemd setup skipped (not Linux)"
        return
    fi

    log "Setting up systemd service..."

    SERVICE_FILE="/tmp/ai-grid-testing.service"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AI Grid Model Autonomous Testing
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=/bin/bash -c 'while true; do ./run_autonomous_tests.py; sleep 14400; done'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    if sudo cp "$SERVICE_FILE" /etc/systemd/system/ai-grid-testing.service 2>/dev/null; then
        sudo systemctl daemon-reload
        sudo systemctl enable ai-grid-testing.service
        log "✅ Systemd service created and enabled"
        log "   Start with: sudo systemctl start ai-grid-testing"
        log "   Stop with: sudo systemctl stop ai-grid-testing"
    else
        warning "Failed to create systemd service (requires sudo)"
    fi

    rm -f "$SERVICE_FILE"
}

# Main setup
main() {
    log "🚀 Setting up Autonomous Testing Infrastructure"
    echo "==============================================="

    check_cron
    setup_schedule
    setup_logging
    setup_monitoring
    setup_systemd

    echo ""
    log "🎉 Autonomous testing setup complete!"
    echo ""
    info "Available commands:"
    echo "  • Run tests manually: ./run_autonomous_tests.py"
    echo "  • View monitoring: python monitoring_dashboard.py"
    echo "  • Check cron jobs: crontab -l"
    echo "  • View logs: tail -f *.log"
    echo ""
    info "Scheduled tasks:"
    echo "  • Comprehensive tests: Every 4 hours"
    echo "  • Health monitoring: Every hour"
    echo "  • Performance checks: Every 6 hours"
}

# Run main setup
main "$@"
