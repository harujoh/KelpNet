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
    public class Sinh<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Sinh";

        public Sinh(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Sinh<float> sinhF:
                    sinhF.SingleInputForward = (x) => SinhF.SingleInputForward(x, sinhF);
                    sinhF.SingleOutputBackward = SinhF.SingleOutputBackward;
                    break;
                case Sinh<double> sinhD:
                    sinhD.SingleInputForward = (x) => SinhD.SingleInputForward(x, sinhD);
                    sinhD.SingleOutputBackward = SinhD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SinhD
#else
    public static class SinhF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> sinh)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Sinh(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, sinh);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += Math.Cosh(x.Data[i]) * y.Grad[i];
            }
        }
    }
}
