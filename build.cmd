@echo off

@echo [build.cmd] build machinelearning
cd machinelearning
if "%1"=="ml" goto compileml:
if exist bin\x64.Release goto mldeb:
:compileml:
cmd /C build.cmd -release
:mldeb:
if "%1"=="ml" goto compilemld:
if exist bin\x64.Debug goto mlrel:
:compilemld:
cmd /C build.cmd -debug
:mlrel:
cd ..

if "%1"=="ml" goto docopy:
if not exist machinelearning\bin\x64.Debug goto end:

:docopy:
@echo [build.cmd] Publish Release
if not exist machinelearning\dist\Release mkdir machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Ensemble\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.FastTree\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.KMeansClustering\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.LightGBM\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.Onnx\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.PCA\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\AnyCPU.Release\Microsoft.ML.StandardLearners\netstandard2.0\*.dll machinelearning\dist\Release
copy machinelearning\bin\x64.Release\Native\*.dll machinelearning\dist\Release

@echo [build.cmd] Publish Debug
if not exist machinelearning\dist\Debug mkdir machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Api\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Maml\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Ensemble\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.FastTree\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.KMeansClustering\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.LightGBM\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.Onnx\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.PCA\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\AnyCPU.Debug\Microsoft.ML.StandardLearners\netstandard2.0\*.dll machinelearning\dist\Debug
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearning\dist\Debug

@echo [build.cmd] build machinelearningext
cd machinelearningext
cmd /C dotnet build -c Debug
cmd /C dotnet build -c Release
cd ..

:finalcopy:
copy machinelearning\bin\x64.Release\Native\*.dll machinelearningext\TestProfileBenchmark\bin\Release\netcoreapp2.1
copy machinelearning\bin\x64.Debug\Native\*.dll machinelearningext\TestProfileBenchmark\bin\Debug\netcoreapp2.1

:end:
if not exist machinelearning\bin\x64.Debug @echo [build.cmd] Cannot build.
@echo [build.cmd] Completed.

