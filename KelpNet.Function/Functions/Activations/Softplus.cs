using System;
using System.Runtime.Serialization;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class Softplus<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Softplus";

        public T Beta;
        public T BetaInv;

        public Softplus(double beta = 1, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            switch (this)
            {
                case Softplus<float> softplusF:
                    softplusF.Beta = (float)beta;
                    softplusF.BetaInv = 1.0f / (float)beta;
                    break;

                case Softplus<double> softplusD:
                    softplusD.Beta = beta;
                    softplusD.BetaInv = 1.0 / beta;
                    break;
            }

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
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real Beta, Real BetaInv, IFunction<Real> softplus)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * Beta;
                }

                Real maxval = y[b * x.Length];
                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (maxval + KelpMath.Log(1.0f + KelpMath.Exp(-KelpMath.Abs(x.Data[i + b * x.Length] * Beta)))) * BetaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, softplus);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, Real Beta)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (1.0f - 1.0f / (1.0f + KelpMath.Exp(Beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

