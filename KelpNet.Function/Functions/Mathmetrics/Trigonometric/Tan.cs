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
    public class Tan<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Tan";

        public Tan(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Tan<float> tanF:
                    tanF.SingleInputForward = (x) => TanF.SingleInputForward(x, tanF);
                    tanF.SingleOutputBackward = TanF.SingleOutputBackward;
                    break;
                case Tan<double> tanD:
                    tanD.SingleInputForward = (x) => TanD.SingleInputForward(x, tanD);
                    tanD.SingleOutputBackward = TanD.SingleOutputBackward;
                    break;
            }
        }

    }
#endif

#if DOUBLE
    public static class TanD
#else
    public static class TanF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> tan)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = Math.Tan(x.Data[i]);
            }

            return new NdArray<Real>(resultData, x.Shape, x.BatchCount, tan);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = Math.Cos(x.Data[i]);
                x.Grad[i] += 1 / (gx * gx) * y.Grad[i];
            }
        }
    }
}
