using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace KelpNet
{
    public static class ArrayConverter
    {
        public static Array ToNdArray<T>(this IEnumerable<T> iEnum, params int[] shape) where T : unmanaged, IComparable<T>
        {
            T[] array = iEnum.ToArray();
            Array result = Array.CreateInstance(array.GetType().GetElementType(), shape);

#if DEBUG
            if (array.Length != result.Length) throw new Exception();
#endif

            Buffer.BlockCopy(array, 0, result, 0, Marshal.SizeOf(array.GetType().GetElementType()) * array.Length);

            return result;
        }

        public static T[] Flatten<T>(this T[,] data) where T : unmanaged, IComparable<T>
        {
            return FlattenEx<T>(data);
        }

        public static T[] Flatten<T>(this T[,,] data) where T : unmanaged, IComparable<T>
        {
            return FlattenEx<T>(data);
        }

        public static T[] Flatten<T>(this T[,,,] data) where T : unmanaged, IComparable<T>
        {
            return FlattenEx<T>(data);
        }

        public static T[] FlattenEx<T>(this Array data) where T : unmanaged, IComparable<T>
        {
            Type arrayType = data.GetType().GetElementType();
            T[] result = new T[data.Length];

            if (typeof(T) == arrayType)
            {
                //タイプ一致なら　一次元化のみ
                Buffer.BlockCopy(data, 0, result, 0, Marshal.SizeOf(arrayType) * data.Length);
            }
            else if (data.Rank == 1)
            {
                //型違いで一次元なら型変換しつつコピーのみ
                Array.Copy(data, result, data.Length);
            }
            else
            {
                //元の型で一次元の長さの配列を用意
                Array array = Array.CreateInstance(arrayType, data.Length);

                //一次元化して
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * data.Length);

                //型変換しつつコピー
                Array.Copy(array, result, array.Length);
            }

            return result;
        }
    }
}
