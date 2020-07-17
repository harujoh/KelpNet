using System;
using System.Collections;
using System.Linq;
using System.Runtime.InteropServices;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
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

        public static void InitXavier<T>(NdArray<T> array) where T : unmanaged, IComparable<T>
        {
            switch (array)
            {
                case NdArray<float> arrayF:
                    InitializerF.InitXavier(arrayF);
                    break;

                case NdArray<double> arrayD:
                    InitializerD.InitXavier(arrayD);
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
        public static void InitXavier(NdArray<Real> array)
        {
            Real fanOut = array.Shape.Length > 1 ? array.Shape[array.Shape.Length - 2] : array.Shape[array.Shape.Length - 1];
            Real fanIn = array.Shape[array.Shape.Length - 1];

            Real n = (fanIn + fanOut) / 2.0f;

            Real limit = Math.Sqrt(3.0f / n);

            for (int i = 0; i < array.Data.Length; i++)
            {
                array.Data[i] = (limit * 2.0f) * Broth.Random() - limit;
            }
        }

        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitHeNorm(NdArray<Real> array, Real masterScale = 1)
        {
            Real s = masterScale * Math.Sqrt(2.0f / array.Length);

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
