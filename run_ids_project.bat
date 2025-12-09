@echo off
REM ===== Intrusion Detection System + Results Analysis =====

REM 1) الانتقال إلى مجلد المشروع 
cd /d "C:\Users\zyadd\Desktop\hail university\1) Network Security\project-for-students"

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
echo Done. Press any key to exit...
pause >nul
