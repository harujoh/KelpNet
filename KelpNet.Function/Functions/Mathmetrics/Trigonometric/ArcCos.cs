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
    public class ArcCos<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "ArcCos";

        public ArcCos(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ArcCos<float> arcCosF:
                    arcCosF.SingleInputForward = (x) => ArcCosF.SingleInputForward(x, arcCosF);
                    arcCosF.SingleOutputBackward = ArcCosF.SingleOutputBackward;
                    break;
                case ArcCos<double> arcCosD:
                    arcCosD.SingleInputForward = (x) => ArcCosD.SingleInputForward(x, arcCosD);
                    arcCosD.SingleOutputBackward = ArcCosD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class ArcCosD
#else
    public static class ArcCosF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> arcCos)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Acos(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, arcCos);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += -1 / Math.Sqrt(-x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
