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
    [Serializable]
    public class ELU<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ELU";

        public T Alpha;

        public ELU(T? alpha = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Alpha = alpha??(TVal<T>)1;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ELU<float> eluF:
                    eluF.SingleInputForward = (x) => ELUF.SingleInputForward(x, eluF.Alpha, eluF);
                    eluF.SingleOutputBackward = (y, x) => ELUF.SingleOutputBackward(y, x, eluF.Alpha);
                    break;

                case ELU<double> eluD:
                    eluD.SingleInputForward = (x) => ELUD.SingleInputForward(x, eluD.Alpha, eluD);
                    eluD.SingleOutputBackward = (y, x) => ELUD.SingleOutputBackward(y, x, eluD.Alpha);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class ELUD
#else
    public static class ELUF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real alpha, IFunction<Real> elu)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    result[i] = x.Data[i];
                }
                else
                {
                    result[i] = alpha * (Math.Exp(x.Data[i]) - 1);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, elu);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, Real alpha)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    x.Grad[i] += y.Grad[i];
                }
                else
                {
                    x.Grad[i] += y.Grad[i] * alpha * Math.Exp(x.Data[i]);
                }
            }
        }
    }
}
