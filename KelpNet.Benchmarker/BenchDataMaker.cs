using System;

namespace KelpNet.Benchmark
{
    class BenchDataMaker<T> where T : unmanaged, IComparable<T>
    {
        public static RealArray<T> GetArray(int length)
        {
            RealArray<T> result = new T[length];

            for (int i = 0; i < length; i++)
            {
                result[i] = Mother<T>.Next();
            }

            return result;
        }
    }
}
