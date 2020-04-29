using System;

#if !NETSTANDARD2_1
namespace KelpNet
{
    public static class MathF
    {
        //public const float E = 2.71828183f;

        public const float PI = 3.14159265f;

        public static float Sqrt(float x)
        {
            return (float)Math.Sqrt(x);
        }

        public static float Log(float x)
        {
            return (float)Math.Log(x);
        }

        public static float Log(float x, float y)
        {
            return (float)Math.Log(x, y);
        }

        public static float Exp(float x)
        {
            return (float)Math.Exp(x);
        }

        public static float Pow(float x, float y)
        {
            return (float)Math.Pow(x, y);
        }

        public static float Asin(float x)
        {
            return (float)Math.Asin(x);
        }

        public static float Sin(float x)
        {
            return (float)Math.Sin(x);
        }

        public static float Acos(float x)
        {
            return (float)Math.Acos(x);
        }

        public static float Cos(float x)
        {
            return (float)Math.Cos(x);
        }

        public static float Cosh(float x)
        {
            return (float)Math.Cosh(x);
        }

        public static float Atan(float x)
        {
            return (float)Math.Atan(x);
        }

        public static float Tan(float x)
        {
            return (float)Math.Tan(x);
        }

        public static float Tanh(float x)
        {
            return (float)Math.Tanh(x);
        }

        public static float Floor(float x)
        {
            return (float)Math.Floor(x);
        }

        public static float Ceiling(float x)
        {
            return (float)Math.Ceiling(x);
        }

        public static float Round(float x)
        {
            return (float)Math.Round(x);
        }

        public static float Max(float x, float y)
        {
            return Math.Max(x, y);
        }

        public static float Min(float x, float y)
        {
            return Math.Min(x, y);
        }

        public static float Abs(float x)
        {
            return Math.Abs(x);
        }
    }
}
#endif
