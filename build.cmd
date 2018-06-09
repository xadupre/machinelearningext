@echo off

@echo [build.cmd] build machinelearning
cd machinelearning
if exist bin\x64.Release goto mldeb:
cmd /C build.cmd -release
:mldeb:
if exist bin\x64.Debug goto mlrel:
cmd /C build.cmd -debug
:mlrel:
cd ..

if not exist machinelearning\bin\x64.Debug goto end:

@echo [build.cmd] Publish
cd machinelearning
dotnet publish Microsoft.ML.sln -o ..\..\dist\Debug -c Debug --self-contained
dotnet publish Microsoft.ML.sln -o ..\..\dist\Release -c Release --self-contained
cd ..

@echo [build.cmd] copy Native DLL
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearning\dist\Debug
copy machinelearning\bin\x64.Release\Native\*.dll machinelearning\dist\Release

@echo [build.cmd] build machinelearningext
cd machinelearningext
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

