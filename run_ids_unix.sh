#!/bin/bash
# ===== Intrusion Detection System + Results Analysis =====

echo ""
echo "[STEP 1] Running IDS.py on data.csv ..."
python3 IDS.py -csv data.csv

echo ""
echo "[STEP 2] Running analyze_results.py to generate charts ..."
python3 analyze_results.py

echo ""
echo "[STEP 3] Running export_results_to_csv.py to generate numbers ..."
python3 export_results_to_csv.py

echo ""
echo "[STEP 4] Running Visualize_Results.py to generate charts ..."
python3 Visualize_Results.py -results data.csv_50.results.txt

echo ""
echo "Done. Press Enter to exit..."
read -p ""