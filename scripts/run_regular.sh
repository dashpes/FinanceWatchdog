#!/bin/bash
# run_regular.sh - Regular data collection and immediate alerts
# Runs every 4 hours during market days

set -e

cd /app

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting regular data collection..."

python -m investment_monitor.cli --type regular

echo "$(date '+%Y-%m-%d %H:%M:%S') - Regular data collection completed."
