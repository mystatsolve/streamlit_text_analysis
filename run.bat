@echo off
chcp 65001 >nul
echo ========================================
echo  텍스트 빈도 분석 도구 시작
echo ========================================
echo.
echo 브라우저가 자동으로 열립니다.
echo 종료하려면 이 창을 닫거나 Ctrl+C 를 누르세요.
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
call C:\Users\user\anaconda3\Scripts\activate.bat py310_2
streamlit run "%~dp0app.py" --server.port 8501 --browser.gatherUsageStats false
pause
