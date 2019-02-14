using System;
using System.Runtime.InteropServices;
using System.Security;
//using RealType = System.Double;
using RealType = System.Single;

namespace KelpNet
{
    public class RealTool
    {
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);
    }

    [Serializable]
    public struct Real : IComparable<Real>
    {
        public readonly RealType Value;

        public static int Size => sizeof(RealType);
        public static Type Type => typeof(RealType);

        private Real(RealType value)
        {
            this.Value = value;
        }

        public static implicit operator Real(double value)
        {
            return new Real((RealType)value);
        }

        public static implicit operator Real(float value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        public int CompareTo(Real other)
        {
            return this.Value.CompareTo(other.Value);
        }

        public override string ToString()
        {
            return this.Value.ToString();
        }

        public static RealType[] ToBaseArray(Array array)
        {
            RealType[] result = new RealType[array.Length];

            //データを叩き込む
            GCHandle source = GCHandle.Alloc(array, GCHandleType.Pinned);

            Marshal.Copy(source.AddrOfPinnedObject(), result, 0, array.Length);            

            source.Free();

            return result;
        }

        public static Array ToBaseNdArray(Array data)
        {
            int[] shape = new int[data.Rank];

            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = data.GetLength(i);
            }

            Array result = Array.CreateInstance(typeof(RealType), shape);

            //データを叩き込む
            GCHandle source = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle dest = GCHandle.Alloc(result, GCHandleType.Pinned);
            RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), sizeof(RealType) * data.Length);
            source.Free();
            dest.Free();

            return result;
        }

        public static Array ToRealNdArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();

#if DEBUG
            //そもそもRealTypeなら必要ない
            if (arrayType == typeof(Real)) throw new Exception();
#endif

            int[] shape = new int[data.Rank];

            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = data.GetLength(i);
            }

            Array resultData = Array.CreateInstance(typeof(Real), shape);

            //型の不一致をここで吸収
            if (arrayType != typeof(RealType))
            {
                //入力と同じ次元の配列を用意
                Array array = Array.CreateInstance(typeof(RealType), shape);

                //型変換しつつコピー
                Array.Copy(data, array, array.Length);

                //データを叩き込む
                GCHandle source = GCHandle.Alloc(array, GCHandleType.Pinned);
                GCHandle dest = GCHandle.Alloc(resultData, GCHandleType.Pinned);
                RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), sizeof(RealType) * data.Length);
                source.Free();
                dest.Free();
            }
            else
            {
                //データを叩き込む
                GCHandle source = GCHandle.Alloc(data, GCHandleType.Pinned);
                GCHandle dest = GCHandle.Alloc(resultData, GCHandleType.Pinned);
                RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), sizeof(RealType) * data.Length);
                source.Free();
                dest.Free();
            }

            return resultData;
        }

        public static Real[] ToRealArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();
            Real[] resultData = new Real[data.Length];

            //型の不一致をここで吸収
            if (arrayType != typeof(RealType) && arrayType != typeof(Real))
            {
                //一次元の長さの配列を用意
                Array array = Array.CreateInstance(arrayType, data.Length);
                //一次元化して
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * resultData.Length);

                data = new RealType[array.Length];

                //型変換しつつコピー
                Array.Copy(array, data, array.Length);
            }

            //データを叩き込む
            GCHandle source = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle dest = GCHandle.Alloc(resultData, GCHandleType.Pinned);
            RealTool.CopyMemory(dest.AddrOfPinnedObject(), source.AddrOfPinnedObject(), sizeof(RealType) * data.Length);
            source.Free();
            dest.Free();

            return resultData;
        }
    }
}
