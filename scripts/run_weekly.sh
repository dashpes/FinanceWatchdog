#!/bin/bash
# run_weekly.sh - Weekly synthesis generation
# Runs every Sunday at 6pm

set -e

cd /app

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting weekly synthesis generation..."

python -m investment_monitor.cli --type weekly

echo "$(date '+%Y-%m-%d %H:%M:%S') - Weekly synthesis generation completed."
