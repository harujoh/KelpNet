using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace KelpNet
{
    [Serializable]
    public struct Real<T> : IComparable<Real<T>> where T : unmanaged, IComparable<T>
    {
        private T _value;

        //Add
        public static T operator +(T x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<T, float>(ref x) + Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<T, double>(ref x) + Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static T operator +(Real<T> x, T y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) + Unsafe.As<T, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) + Unsafe.As<T, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static Real<T> operator +(Real<T> x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) + Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, Real<T>>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) + Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, Real<T>>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        //Sub
        public static T operator -(T x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<T, float>(ref x) - Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<T, double>(ref x) - Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static T operator -(Real<T> x, T y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) - Unsafe.As<T, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) - Unsafe.As<T, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static Real<T> operator -(Real<T> x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) - Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, Real<T>>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) - Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, Real<T>>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        //Mul
        public static T operator *(T x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<T, float>(ref x) * Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<T, double>(ref x) * Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static T operator *(Real<T> x, T y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) * Unsafe.As<T, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) * Unsafe.As<T, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static Real<T> operator *(Real<T> x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) * Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, Real<T>>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) * Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, Real<T>>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        //Div
        public static T operator /(T x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<T, float>(ref x) / Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<T, double>(ref x) / Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static T operator /(Real<T> x, T y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) / Unsafe.As<T, float>(ref y);
                return Unsafe.As<float, T>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) / Unsafe.As<T, double>(ref y);
                return Unsafe.As<double, T>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        public static Real<T> operator /(Real<T> x, Real<T> y)
        {
            if (typeof(T) == typeof(float))
            {
                float result = Unsafe.As<Real<T>, float>(ref x) / Unsafe.As<Real<T>, float>(ref y);
                return Unsafe.As<float, Real<T>>(ref result);
            }
            else if (typeof(T) == typeof(double))
            {
                double result = Unsafe.As<Real<T>, double>(ref x) / Unsafe.As<Real<T>, double>(ref y);
                return Unsafe.As<double, Real<T>>(ref result);
            }
            else
            {
                throw new Exception();
            }
        }

        //Cast
        public static implicit operator Real<T>(double real)
        {
            if (typeof(T) == typeof(double))
            {
                return Unsafe.As<double, Real<T>>(ref real);
            }
            else if (typeof(T) == typeof(float))
            {
                float val = (float)real;
                return Unsafe.As<float, Real<T>>(ref val);
            }

            throw new Exception();
        }

        public static implicit operator Real<T>(T real)
        {
            return Unsafe.As<T, Real<T>>(ref real);
        }

        public static implicit operator double(Real<T> real)
        {
            if (typeof(T) == typeof(double))
            {
                return Unsafe.As<Real<T>, double>(ref real);
            }
            else if (typeof(T) == typeof(float))
            {
                return Unsafe.As<Real<T>, float>(ref real);
            }
            else
            {
                throw new Exception();
            }
        }

        public static implicit operator float(Real<T> real)
        {
            if (typeof(T) == typeof(double))
            {
                double val = Unsafe.As<Real<T>, double>(ref real);
                return (float)val;
            }
            else if (typeof(T) == typeof(float))
            {
                return Unsafe.As<Real<T>, float>(ref real);
            }
            else
            {
                throw new Exception();
            }
        }

        public static unsafe Real<T>[] GetArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();
            Real<T>[] resultData = new Real<T>[data.Length];

            //型の不一致をここで吸収
            if (arrayType != typeof(T) || arrayType != typeof(Real<T>))
            {
                //一次元の長さの配列を用意
                Array array = Array.CreateInstance(arrayType, data.Length);
                //一次元化して
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * data.Length);

                data = new T[array.Length];

                //型変換しつつコピー
                Array.Copy(array, data, array.Length);
            }

            //データを叩き込む
            int size = sizeof(T) * data.Length;
            GCHandle gchObj = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle gchBytes = GCHandle.Alloc(resultData, GCHandleType.Pinned);
            Buffer.MemoryCopy((void*)gchObj.AddrOfPinnedObject(), (void*)gchBytes.AddrOfPinnedObject(), size, size);
            gchObj.Free();
            gchBytes.Free();

            return resultData;
        }

        public int CompareTo(Real<T> other)
        {
            return this._value.CompareTo(other._value);
        }

        public override string ToString()
        {
            return this._value.ToString();
        }
    }
}
