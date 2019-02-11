using System;

namespace KelpNet.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            //単精度ベンチマーク
            Console.WriteLine("Start Single...\n");
            CpuBenchmarker<float>.Run();

            //倍精度ベンチマーク
            //Console.WriteLine("\nStart Double...\n");
            //CpuBenchmarker<double>.Run();

            Console.WriteLine("Done...");
            Console.Read();
        }
    }
}
