using System;

namespace TestProfileBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var df = Benchmark_PredictionEngine.TestScikitAPI_EngineSimpleTrainAndPredict("mlnet", 1);
            Console.WriteLine(df.ToString());
        }
    }
}
