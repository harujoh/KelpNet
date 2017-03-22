using KelpNet.Common;

namespace KelpNetTester.Benchmarker
{
    class BenchDataMaker
    {
        public static double[] GetDoubleArray(int length)
        {
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Mother.Dice.NextDouble();
            }

            return result;
        }
    }
}
