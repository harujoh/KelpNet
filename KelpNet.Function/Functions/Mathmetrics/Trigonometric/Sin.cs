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
    public class Sin<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Sin";

        public Sin(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Sin<float> sinF:
                    sinF.SingleInputForward = (x) => SinF.SingleInputForward(x, sinF);
                    sinF.SingleOutputBackward = SinF.SingleOutputBackward;
                    break;
                case Sin<double> sinD:
                    sinD.SingleInputForward = (x) => SinD.SingleInputForward(x, sinD);
                    sinD.SingleOutputBackward = SinD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SinD
#else
    public static class SinF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> sin)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Sin(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, sin);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += Math.Cos(x.Data[i]) * y.Grad[i];
            }
        }
    }
}
