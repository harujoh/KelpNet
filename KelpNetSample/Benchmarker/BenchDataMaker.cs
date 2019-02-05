using KelpNet.Common;

namespace KelpNetSample.Benchmarker
{
    class BenchDataMaker
    {
        public static Real[] GetRealArray(int length)
        {
            Real[] result = new Real[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Mother.Dice.NextDouble();
            }

            return result;
        }
    }
}
