
namespace KelpNet.Benchmark
{
    class BenchDataMaker
    {
        public static float[] GetFloatArray(int length)
        {
            float[] result = new float[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (float)Mother.Dice.NextDouble();
            }

            return result;
        }

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
