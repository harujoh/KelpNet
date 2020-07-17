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
    public class Softplus<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Softplus";

        public T Beta;
        public T BetaInv;

        public Softplus(T? beta = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Beta = beta??(TVal<T>)1.0;
            this.BetaInv = (TVal<T>)1.0 / this.Beta;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Softplus<float> softplusF:
                    softplusF.SingleInputForward = (x) => SoftplusF.SingleInputForward(x, softplusF.Beta, softplusF.BetaInv, softplusF);
                    softplusF.SingleOutputBackward = (y, x) => SoftplusF.SingleOutputBackward(y, x, softplusF.Beta);
                    break;

                case Softplus<double> softplusD:
                    softplusD.SingleInputForward = (x) => SoftplusD.SingleInputForward(x, softplusD.Beta, softplusD.BetaInv, softplusD);
                    softplusD.SingleOutputBackward = (y, x) => SoftplusD.SingleOutputBackward(y, x, softplusD.Beta);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SoftplusD
#else
    public static class SoftplusF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real beta, Real betaInv, IFunction<Real> softplus)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * beta;
                }

                //0との比較が必要
                Real maxval = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (maxval + Math.Log(1.0f + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * beta)))) * betaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, softplus);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, Real beta)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (1.0f - 1.0f / (1.0f + Math.Exp(beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

