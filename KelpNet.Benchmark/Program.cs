using System;

namespace KelpNet.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            //ベンチマーク
            CpuBenchmarker.Run();

            Console.WriteLine("Done...");
            Console.Read();
        }
    }
}
