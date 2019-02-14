using System;
using System.Runtime.InteropServices;

namespace KelpNet
{
    public class Initializer
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray array, double masterScale = 1.0)
        {
            double s = 1.0 / Math.Sqrt(2) * Math.Sqrt(2.0 / array.Length);

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Mother.RandomNormal(s) * masterScale;
            }
        }

        //適当な値の配列を作る
        public static Real[] GetRealArray(int count, int max = 1, int min = 0)
        {
            Real[] result = new Real[count];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (max - min) * Mother.Dice.NextDouble() - min;
            }

            return result;
        }

        //多次元配列を作る
        public static Array GetRealNdArray(int[] shape, int max = 1, int min = 0)
        {
            Array result = Array.CreateInstance(typeof(Real), shape);

            Real[] tmp = new Real[result.Length];

            for (int i = 0; i < tmp.Length; i++)
            {
                tmp[i] = (max - min) * Mother.Dice.NextDouble() - min;
            }

            GCHandle source = GCHandle.Alloc(tmp, GCHandleType.Pinned);
            GCHandle dest = GCHandle.Alloc(result, GCHandleType.Pinned);

            RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), result.Length * Real.Size);

            dest.Free();
            source.Free();

            return result;
        }

    }
}
