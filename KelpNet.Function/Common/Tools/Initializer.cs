using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public static class Initializer
    {
        public static void InitWeight<T>(NdArray<T> array) where T : unmanaged, IComparable<T>
        {
            switch (array)
            {
                case NdArray<float> arrayF:
                    InitializerF.InitWeight(arrayF);
                    break;

                case NdArray<double> arrayD:
                    InitializerD.InitWeight(arrayD);
                    break;
            }
        }

        ////適当な値の配列を作る
        //public static T[] GetRealArray<T>(int count, Real max = 1, Real min = 0) where T : unmanaged, IComparable<T>
        //{
        //    T[] result = new T[count];

        //    switch (result)
        //    {
        //        case float[] resultF:
        //            InitializerF.GetRealArray(resultF, max, min);
        //            break;

        //        case double[] resultD:
        //            InitializerD.GetRealArray(resultD, max, min);
        //            break;
        //    }

        //    return result;
        //}

        //多次元配列を作る
        public static T GetRandomValues<T>(int[] shape, Real max, Real min = 0) where T : ICollection, IEnumerable, ICloneable, IList, IStructuralComparable, IStructuralEquatable
        {
            Array result = Array.CreateInstance(typeof(T).GetElementType(), shape);

            Array tmp = Array.CreateInstance(typeof(T).GetElementType(), result.Length);

            switch (tmp)
            {
                case float[] tmpF:
                    InitializerF.GetRealArray(tmpF, max, min);
                    break;

                case double[] tmpD:
                    InitializerD.GetRealArray(tmpD, max, min);
                    break;
            }

            Buffer.BlockCopy(tmp, 0, result, 0, result.Length * Marshal.SizeOf(typeof(T).GetElementType()));

            return (T)(object)result;
        }

        public static T GetRandomValues<T>(params int[] shape) where T : ICollection, IEnumerable, ICloneable, IList, IStructuralComparable, IStructuralEquatable
        {
            return GetRandomValues<T>(shape, 1);
        }
    }


#endif

#if DOUBLE
    public static class InitializerD
#else
    public static class InitializerF
#endif
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray<Real> array, Real masterScale = 1)
        {
            Real s = masterScale * KelpMath.Sqrt(2.0f / array.Length);

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = Broth.RandomNormal(s);
            }
        }

        //適当な値の配列を作る
        public static void GetRealArray(Real[] result, Real max = 1, Real min = 0)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (max - min) * Broth.Random() - min;
            }
        }

        //適当な値の配列を作る
        //public static Real[] GetRealArray(int count, int max = 1, int min = 0)
        //{
        //    Real[] result = new Real[count];

        //    for (int i = 0; i < result.Length; i++)
        //    {
        //        result[i] = (max - min) * Mother.Dice.NextDouble() - min;
        //    }

        //    return result;
        //}

        ////多次元配列を作る
        //public static Array GetRealNdArray(int[] shape, int max = 1, int min = 0)
        //{
        //    Array result = Array.CreateInstance(typeof(Real), shape);

        //    Real[] tmp = new Real[result.Length];

        //    for (int i = 0; i < tmp.Length; i++)
        //    {
        //        tmp[i] = (max - min) * Mother.Dice.NextDouble() - min;
        //    }

        //    GCHandle source = GCHandle.Alloc(tmp, GCHandleType.Pinned);
        //    GCHandle dest = GCHandle.Alloc(result, GCHandleType.Pinned);

        //    RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), result.Length * Real.Size);

        //    dest.Free();
        //    source.Free();

        //    return result;
        //}

        //範囲指定の配列を作る
        //public static Array Range(int[] shape, int start = 0)
        //{
        //    int count = 0;

        //    if (count == 0)
        //    {
        //        count = shape[0];

        //        for (int i = 1; i < shape.Length; i++)
        //        {
        //            count *= shape[i];
        //        }
        //    }

        //    return Real.ToRealNdArray(Enumerable.Range(start, count).ToNdArray(shape));
        //}

        //        [Suppressunmanaged, IComparable<T>CodeSecurity]
        //        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        //        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);

        //        public static Array ToNdArray<T>(this IEnumerable<T> iEnum, params int[] shape)
        //        {
        //            Array array = iEnum.ToArray();
        //            Array result = Array.CreateInstance(array.GetType().GetElementType(), shape);

        //#if DEBUG
        //            if (array.Length != result.Length) throw new Exception();
        //#endif

        //            GCHandle source = GCHandle.Alloc(array, GCHandleType.Pinned);
        //            GCHandle dest = GCHandle.Alloc(result, GCHandleType.Pinned);

        //            CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), Marshal.SizeOf(array.GetType().GetElementType()) * array.Length);

        //            dest.Free();
        //            source.Free();

        //            return result;
        //        }
    }
}
