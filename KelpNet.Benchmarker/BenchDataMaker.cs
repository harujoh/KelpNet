using System;

namespace KelpNet.Benchmark
{
    class BenchDataMaker<T> where T : unmanaged, IComparable<T>
    {
        public static Real<T>[] GetArray(int length)
        {
            Real<T>[] result = new Real<T>[length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Mother<T>.Next();
            }

            return result;
        }
    }
}
