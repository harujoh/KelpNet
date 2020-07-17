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
    public class ArcTan<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "ArcTan";

        public ArcTan(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ArcTan<float> arcTanF:
                    arcTanF.SingleInputForward = (x) => ArcTanF.SingleInputForward(x, arcTanF);
                    arcTanF.SingleOutputBackward = ArcTanF.SingleOutputBackward;
                    break;
                case ArcTan<double> arcTanD:
                    arcTanD.SingleInputForward = (x) => ArcTanD.SingleInputForward(x, arcTanD);
                    arcTanD.SingleOutputBackward = ArcTanD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class ArcTanD
#else
    public static class ArcTanF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> arcTan)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Atan(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, arcTan);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += 1 / (x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
