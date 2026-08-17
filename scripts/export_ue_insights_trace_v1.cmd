@echo off
setlocal

if "%~4"=="" (
  echo Usage: %~nx0 TRACE_FILE OUTPUT_FILE LOG_FILE MODE
  echo MODE must be threads, timers, events, or gpu_events.
  exit /b 2
)

set "TRACE_FILE=%~1"
set "OUTPUT_FILE=%~2"
set "LOG_FILE=%~3"
set "MODE=%~4"
if not defined UE_INSIGHTS_EXE (
  echo Set UE_INSIGHTS_EXE to the UE 5.8 UnrealInsights.exe path.
  exit /b 2
)
set "INSIGHTS_EXE=%UE_INSIGHTS_EXE%"
if not exist "%INSIGHTS_EXE%" (
  echo UnrealInsights.exe not found: %INSIGHTS_EXE%
  exit /b 2
)

if /I "%MODE%"=="threads" set "INSIGHTS_COMMAND=TimingInsights.ExportThreads %OUTPUT_FILE%"
if /I "%MODE%"=="timers" set "INSIGHTS_COMMAND=TimingInsights.ExportTimers %OUTPUT_FILE%"
if /I "%MODE%"=="events" set "INSIGHTS_COMMAND=TimingInsights.ExportTimingEvents %OUTPUT_FILE%"
if /I "%MODE%"=="gpu_events" set "INSIGHTS_COMMAND=TimingInsights.ExportTimingEvents %OUTPUT_FILE% -columns=ThreadId,ThreadName,TimerId,TimerName,StartTime,EndTime,Duration,Depth -threads=GPU0-Graphics0 -timers=Frame*,BasePass"

if not defined INSIGHTS_COMMAND (
  echo Unknown mode: %MODE%
  exit /b 2
)

"%INSIGHTS_EXE%" -OpenTraceFile="%TRACE_FILE%" -ABSLOG="%LOG_FILE%" -AutoQuit -NoUI -ExecOnAnalysisCompleteCmd="%INSIGHTS_COMMAND%" -log
exit /b %ERRORLEVEL%
