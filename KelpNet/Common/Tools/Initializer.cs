using System;

namespace KelpNet.Common.Tools
{
    class Initializer
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray array, double masterScale = 1.0)
        {
            double localScale = 1 / Math.Sqrt(2);
            int fanIn = GetFans(array.Shape);
            double s = localScale * Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < array.Length; i++)
            {
                array.Data[i] = Normal(s) * masterScale;
            }
        }

        private static double Normal(double scale = 0.05)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        private static int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
}
