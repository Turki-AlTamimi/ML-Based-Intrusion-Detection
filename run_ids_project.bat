@echo off



echo.
echo [STEP 1] Running IDS.py on data.csv ...
python IDS.py -csv data.csv

echo.
echo [STEP 2] Running analyze_results.py to generate charts ...
python analyze_results.py

echo.
echo [STEP 3] Running export_results_to_csv.py to generate numbers ...
python export_results_to_csv.py

echo.
echo [STEP 4] RunningVisualize_Results.py to generate charts ...
python Visualize_Results.py -results data.csv_50.results.txt


echo.
echo Done. Press any key to exit...
pause >nul




