#!/bin/bash
# run_digest.sh - Daily digest generation
# Runs every day at 7am

set -e

cd /app

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting daily digest generation..."

python -m investment_monitor.cli --type digest

echo "$(date '+%Y-%m-%d %H:%M:%S') - Daily digest generation completed."
