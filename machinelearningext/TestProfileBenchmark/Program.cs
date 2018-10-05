using System;

namespace TestProfileBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            int N = 2000;
            int ncall = 1;
            for (int th = 1; th <= 1; ++th)
            {
                foreach (var cache in new[] { false, true })
                {
                    Console.WriteLine("");
                    Console.WriteLine("Test scikit...");
                    var df1 = Benchmark_PredictionEngine.TestScikitAPI_EngineSimpleTrainAndPredict("scikit", th, N, ncall, cache);
                    Console.WriteLine(df1.ToString());
                    Console.WriteLine("");
                    Console.WriteLine("Test mlnet...");
                    var df2 = Benchmark_PredictionEngine.TestScikitAPI_EngineSimpleTrainAndPredict("mlnet", th, N, ncall, cache);
                    Console.WriteLine(df2.ToString());
                }
            }
        }
    }
}
