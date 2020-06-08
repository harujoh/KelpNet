using System;

namespace KelpNet
{
    //各パラメータのデフォルト値を設定するためのヘルパークラス
    public struct TVal<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }
        static T GenDiv(T x, T y) { return Operator<T>.Divide(x, y); }

        private T Val;

        TVal(T val)
        {
            this.Val = val;
        }

        public static explicit operator TVal<T>(float value)
        {
            if (typeof(T) == typeof(float)) return new TVal<T>((T)(object)value);
            if (typeof(T) == typeof(double)) return new TVal<T>((T)(object)(double)value);
            throw new Exception();
        }

        public static explicit operator TVal<T>(double value)
        {
            if (typeof(T) == typeof(float)) return new TVal<T>((T)(object)(float)value);
            if (typeof(T) == typeof(double)) return new TVal<T>((T)(object)value);
            throw new Exception();
        }

        public static implicit operator TVal<T>(T real)
        {
            return new TVal<T>(real);
        }

        public static implicit operator T(TVal<T> real)
        {
            return real.Val;
        }

        public static TVal<T> operator +(TVal<T> a, TVal<T> b)
        {
            return new TVal<T>(GenAdd(a.Val, b.Val));
        }

        public static TVal<T> operator -(TVal<T> a, TVal<T> b)
        {
            return new TVal<T>(GenSub(a.Val, b.Val));
        }

        public static TVal<T> operator *(TVal<T> a, TVal<T> b)
        {
            return new TVal<T>(GenMul(a.Val, b.Val));
        }

        public static TVal<T> operator /(TVal<T> a, TVal<T> b)
        {
            return new TVal<T>(GenDiv(a.Val, b.Val));
        }
    }
}
