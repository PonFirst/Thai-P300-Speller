@echo off
rem Thai P300 Speller: double-click to start. The steps are in bci2000\windows\menu.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bci2000\windows\menu.ps1" %*
if errorlevel 1 pause
