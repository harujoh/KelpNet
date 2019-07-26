using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace KelpNet
{
    [Serializable]
    [DebuggerDisplay("Length = {Length}")]
    public unsafe class RealArray<T> : IEnumerable<Real<T>> where T : unmanaged, IComparable<T>
    {
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public IntPtr Ptr;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public int Length;

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        private Real<T>[] DebugView
        {
            get { return this.ToArray(); }
        }

        public RealArray(IntPtr ptr, int length)
        {
            this.Ptr = ptr;
            this.Length = length;
        }

        public Real<T> this[int index]
        {
            get
            {
                return Unsafe.Read<T>((void*)(Ptr + index * sizeof(T)));
            }

            set
            {
                Unsafe.Write((void*)(Ptr + index * sizeof(T)), value);
            }
        }

        //public static implicit operator RealArray<T>(IntPtr a)
        //{
        //    return Unsafe.As<IntPtr, RealArray<T>>(ref a);
        //}

        //public static implicit operator RealArray<T>(T[] a)
        //{
        //    return new RealArray<T>((IntPtr)Unsafe.AsPointer(ref a[0]), a.Length);
        //}

        //多次元で入ってきた配列もここで一次元としてメモリに配置される
        public static implicit operator RealArray<T>(Array array)
        {
            Type arrayType = array.GetType().GetElementType();

            if (arrayType != typeof(Real<T>) && arrayType != typeof(T))
            {
                if (array.Rank != 1)
                {
                    //2次元以上なら1次元化

                    //一次元の長さの配列を用意
                    Array tmp = Array.CreateInstance(arrayType, array.Length);

                    //一次元化して
                    Buffer.BlockCopy(array, 0, tmp, 0, Marshal.SizeOf(arrayType) * array.Length);

                    array = new T[tmp.Length];

                    //型変換しつつコピー
                    Array.Copy(tmp, array, tmp.Length);
                }
                else
                {
                    T[] tmp = new T[array.Length];

                    //型変換
                    Array.Copy(array, tmp,  tmp.Length);

                    array = tmp;
                }
            }

            int size = array.Length * sizeof(T);
            IntPtr ptr = Marshal.AllocCoTaskMem(size);

            GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            Buffer.MemoryCopy((void*)handle.AddrOfPinnedObject(), (void*)ptr, size, size);
            handle.Free();

            return new RealArray<T>(ptr, array.Length);
        }

        public RealArray<T> Clone()
        {
            IntPtr result = Marshal.AllocCoTaskMem(Length * sizeof(T));

            Buffer.MemoryCopy((void*)Ptr, (void*)result, Length * sizeof(T), Length * sizeof(T));

            return new RealArray<T>(result, Length);
        }

        public void CopyTo(T[] dest, int sourceStartIndex, int destStartIndex, int length)
        {
            fixed (void* p = &dest[destStartIndex])
            {
                Buffer.MemoryCopy((void*)(Ptr + sourceStartIndex * sizeof(T)), p, length * sizeof(T), length * sizeof(T));
            }
        }

        public void CopyTo(RealArray<T> dest, int sourceStartIndex, int destStartIndex, int length)
        {
            Buffer.MemoryCopy((void*)(Ptr + sourceStartIndex * sizeof(T)), (void*)(dest.Ptr + destStartIndex * sizeof(T)), length * sizeof(T), length * sizeof(T));
        }

        public void CopyFrom(T[] source, int sourceStartIndex, int destStartIndex, int length)
        {
            fixed (void* p = &source[sourceStartIndex])
            {
                Buffer.MemoryCopy(p, (void*)(Ptr + destStartIndex * sizeof(T)), length * sizeof(T), length * sizeof(T));
            }
        }

        public void CopyFrom(RealArray<T> source, int sourceStartIndex, int destStartIndex, int length)
        {
            Buffer.MemoryCopy((void*)(source.Ptr + sourceStartIndex * sizeof(T)), (void*)(Ptr + destStartIndex * sizeof(T)), length * sizeof(T), length * sizeof(T));
        }

        public static implicit operator NdArray<T>(RealArray<T> real)
        {
            return NdArray<T>.Convert(real);
        }

        public int MaxIndex()
        {
            Real<T> dataMax = this[0];
            int dataMaxIndex = 0;

            for (int i = 1; i < this.Length; i++)
            {
                if (dataMax < this[i])
                {
                    dataMax = this[i];
                    dataMaxIndex = i;
                }
            }

            return dataMaxIndex;
        }

        //IEnumerator<T> GetEnumerator()
        //{
        //    return new RealArrayEnumerator<T>(this);
        //}

        public IEnumerator<Real<T>> GetEnumerator()
        {
            return new RealArrayEnumerator(this);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }


        private class RealArrayEnumerator : IEnumerator<Real<T>>
        {
            private readonly RealArray<T> value;
            private int index = -1;

            public RealArrayEnumerator(RealArray<T> value)
            {
                this.value = value;
            }

            public Real<T> Current
            {
                get { return value[index]; }
            }

            object IEnumerator.Current => Current;

            public bool MoveNext()
            {
                index++;
                return index < value.Length;
            }

            public void Reset()
            {
                index = -1;
            }

            public void Dispose()
            {
            }
        }
    }
}
