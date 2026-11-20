@echo off
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\leaklens.exe" (
  "%HERE%.venv\Scripts\leaklens.exe" %*
  exit /b %ERRORLEVEL%
)
if exist "%HERE%src\leaklens\__main__.py" (
  python -m leaklens %*
  exit /b %ERRORLEVEL%
)
echo leaklens is not installed. From this folder run:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -e .
exit /b 1
