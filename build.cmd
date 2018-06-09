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

@echo [build.cmd] Publish Release
if not exist machinelearning\dist\Release mkdir machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\x64.Release\Native\*.dll machinelearning\dist\Release

@echo [build.cmd] Publish Debug
if not exist machinelearning\dist\Debug mkdir machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearning\dist\Debug

@echo [build.cmd] build machinelearningext
cd machinelearningext
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

