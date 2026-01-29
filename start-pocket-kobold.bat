@echo off
cd /d "%~dp0"
set OVA_TTS_ENGINE=pocket_tts
set OVA_LLM_BACKEND=koboldcpp
"C:\Program Files\Git\bin\bash.exe" ova.sh start
pause
