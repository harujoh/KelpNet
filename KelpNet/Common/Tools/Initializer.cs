using System;

namespace KelpNet
{
    class Initializer<T> where T : unmanaged, IComparable<T>
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray<T> array, double masterScale = 1.0)
        {
            Real<T> s = (1.0 / Math.Sqrt(2.0) * Math.Sqrt(2.0 / GetFans(array.Shape)));

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Mother<T>.RandomNormal(s) * masterScale;
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
