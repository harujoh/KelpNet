using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;

namespace KelpNet
{
    public static class Initializer
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray array, double masterScale = 1.0)
        {
            double s = masterScale * Math.Sqrt(2.0 / array.Length);

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Mother.RandomNormal(s);
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

        //範囲指定の配列を作る
        public static Array Range(int[] shape, int start = 0)
        {
            int count = 0;

            if (count == 0)
            {
                count = shape[0];

                for (int i = 1; i < shape.Length; i++)
                {
                    count *= shape[i];
                }
            }

            return Real.ToRealNdArray(Enumerable.Range(start, count).ToNdArray(shape));
        }

        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);

        public static Array ToNdArray<T>(this IEnumerable<T> iEnum, params int[] shape)
        {
            Array array = iEnum.ToArray();
            Array result = Array.CreateInstance(array.GetType().GetElementType(), shape);

#if DEBUG
            if (array.Length != result.Length) throw new Exception();
#endif

            GCHandle source = GCHandle.Alloc(array, GCHandleType.Pinned);
            GCHandle dest = GCHandle.Alloc(result, GCHandleType.Pinned);

            CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), Marshal.SizeOf(array.GetType().GetElementType()) * array.Length);

            dest.Free();
            source.Free();

            return result;
        }
    }
}
