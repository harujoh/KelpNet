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
        public static void InitHeNorm<T>(NdArray<T> array) where T : unmanaged, IComparable<T>
        {
            switch (array)
            {
                case NdArray<float> arrayF:
                    InitializerF.InitHeNorm(arrayF);
                    break;

                case NdArray<double> arrayD:
                    InitializerD.InitHeNorm(arrayD);
                    break;
            }
        }

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
        public static void InitHeNorm(NdArray<Real> array, Real masterScale = 1)
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

        //範囲指定の配列を作る
        public static Array Range(int[] shape, int start = 0)
        {
            return Enumerable.Range(start, NdArray.ShapeToLength(shape)).ToNdArray(shape);
        }
    }
}
