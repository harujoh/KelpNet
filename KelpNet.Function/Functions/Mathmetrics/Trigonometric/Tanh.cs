using System;
using System.Runtime.Serialization;

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
    public class Tanh<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Tanh";

        public Tanh(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Tanh<float> tanhF:
                    tanhF.SingleInputForward = (x) => TanhF.SingleInputForward(x, tanhF);
                    tanhF.SingleOutputBackward = TanhF.SingleOutputBackward;
                    break;
                case Tanh<double> tanhD:
                    tanhD.SingleInputForward = (x) => TanhD.SingleInputForward(x, tanhD);
                    tanhD.SingleOutputBackward = TanhD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class TanhD
#else
    public static class TanhF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> tanh)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Tanh(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, tanh);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * (1 - y.Data[i] * y.Data[i]);
            }
        }
    }
}
