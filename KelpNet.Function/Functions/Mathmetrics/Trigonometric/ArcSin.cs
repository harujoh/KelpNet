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
    public class ArcSin<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "ArcSin";

        public ArcSin(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ArcSin<float> arcSinF:
                    arcSinF.SingleInputForward = (x) => ArcSinF.SingleInputForward(x, arcSinF);
                    arcSinF.SingleOutputBackward = ArcSinF.SingleOutputBackward;
                    break;
                case ArcSin<double> arcSinD:
                    arcSinD.SingleInputForward = (x) => ArcSinD.SingleInputForward(x, arcSinD);
                    arcSinD.SingleOutputBackward = ArcSinD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class ArcSinD
#else
    public static class ArcSinF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> arcSin)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Asin(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, arcSin);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += 1 / Math.Sqrt(-x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
