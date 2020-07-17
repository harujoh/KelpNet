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
    public class Cosh<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Cosh";

        public Cosh(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Cosh<float> coshF:
                    coshF.SingleInputForward = (x) => CoshF.SingleInputForward(x, coshF);
                    coshF.SingleOutputBackward = CoshF.SingleOutputBackward;
                    break;
                case Cosh<double> coshD:
                    coshD.SingleInputForward = (x) => CoshD.SingleInputForward(x, coshD);
                    coshD.SingleOutputBackward = CoshD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class CoshD
#else
    public static class CoshF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> cosh)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Cosh(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, cosh);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += Math.Sinh(x.Data[i]) * y.Grad[i];
            }
        }
    }
}
