@echo off
chcp 65001 >nul
echo ========================================
echo  토픽 분석 도구 (LDA + BERTopic)
echo ========================================
echo.
echo 브라우저가 자동으로 열립니다.
echo 종료하려면 이 창을 닫거나 Ctrl+C 를 누르세요.
echo.
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
call C:\Users\user\anaconda3\Scripts\activate.bat py310_2
streamlit run "%~dp0topic_app.py" --server.port 8503 --browser.gatherUsageStats false
pause
