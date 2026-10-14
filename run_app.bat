@echo off
echo ========================================================
echo   Launching Quantum Fraud Detection Streamlit Platform
echo ========================================================
if exist .venv\Scripts\streamlit.exe (
    .\.venv\Scripts\streamlit.exe run app.py
) else (
    streamlit run app.py
)
pause
