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
    public class Cos<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Cos";

        public Cos(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Cos<float> cosF:
                    cosF.SingleInputForward = (x) => CosF.SingleInputForward(x, cosF);
                    cosF.SingleOutputBackward = CosF.SingleOutputBackward;
                    break;
                case Cos<double> cosD:
                    cosD.SingleInputForward = (x) => CosD.SingleInputForward(x, cosD);
                    cosD.SingleOutputBackward = CosD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class CosD
#else
    public static class CosF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> cos)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Cos(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, cos);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += Math.Sin(x.Data[i]) * -y.Grad[i];
            }
        }
    }
}
