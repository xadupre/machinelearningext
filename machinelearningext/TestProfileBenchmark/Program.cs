using System;

namespace TestProfileBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
#if(DEBUG)
            int N = 10;
#else
            int N = 2000;
#endif
            Console.WriteLine("Test scikit...");
            for (int ncall = 1; ncall <= 2; ++ncall)
            {
                for (int th = 1; th <= 4; ++th)
                {
                    foreach (var cache in new[] { true, false })
                    {
                        var df1 = Benchmark_PredictionEngine.TestScikitAPI_EngineSimpleTrainAndPredict("scikit", th, N, ncall, cache);
                        Console.WriteLine(df1.ToString());
                    }
                }
            }

            Console.WriteLine("");
            Console.WriteLine("Test mlnet...");
            for (int ncall = 1; ncall <= 2; ++ncall)
            {
                for (int th = 1; th <= 4; ++th)
                {
                    foreach (var cache in new[] { true, false })
                    {
                        var df2 = Benchmark_PredictionEngine.TestScikitAPI_EngineSimpleTrainAndPredict("mlnet", th, N, ncall, cache);
                        Console.WriteLine(df2.ToString());
                    }
                }
            }
        }
    }
}
