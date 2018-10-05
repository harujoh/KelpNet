using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    class Initializer
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray array, Real masterScale = 1.0f)
        {
            Real s = (Real)(1.0 / Math.Sqrt(2.0) * Math.Sqrt(2.0 / GetFans(array.Shape)));

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Mother.RandomNormal(s) * masterScale;
            }
        }

        private static double GetFans(int[] shape)
        {
            double result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
}
