@echo off
echo [1/3] Running dataset verification...
python scripts\verify.py
if %errorlevel% neq 0 goto :error
echo.
echo [2/3] Running main pipeline...
python main.py
if %errorlevel% neq 0 goto :error
echo.
echo [3/3] Done. Check outputs/ for results.
goto :end
:error
echo Pipeline failed. Check the error above.
exit /b 1
:end
