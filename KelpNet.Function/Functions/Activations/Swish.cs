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
    public class Swish<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Swish";

        public NdArray<T> Beta;

        public Swish(int[] betaShape, T? beta = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Beta = new NdArray<T>(betaShape);
            this.Beta.Fill(beta ?? (TVal<T>)1.0);

            this.Parameters = new[] { this.Beta };

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Swish<float> swishF:
                    swishF.SingleInputForward = (x) => SwishF.SingleInputForward(x, swishF.Beta, swishF);
                    swishF.SingleOutputBackward = (y, x) => SwishF.SingleOutputBackward(y, x, swishF.Beta);
                    break;

                case Swish<double> swishD:
                    swishD.SingleInputForward = (x) => SwishD.SingleInputForward(x, swishD.Beta, swishD);
                    swishD.SingleOutputBackward = (y, x) => SwishD.SingleOutputBackward(y, x, swishD.Beta);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SwishD
#else
    public static class SwishF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> beta, IFunction<Real> swish)
        {
            Real[] result = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    int offsetedIndex = b * x.Length + i;
                    result[offsetedIndex] = x.Data[offsetedIndex] * (Math.Tanh(x.Data[offsetedIndex] * beta.Data[i] * 0.5f) * 0.5f + 0.5f);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, swish);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> beta)
        {
            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    int offsetedIndex = b * x.Length + i;
                    Real sig = Math.Tanh(beta.Data[i] * x.Data[offsetedIndex] * 0.5f) * 0.5f + 0.5f;
                    Real by = beta.Data[i] * x.Data[offsetedIndex] * sig;

                    x.Grad[offsetedIndex] += y.Grad[offsetedIndex] * (by + sig * (1 - by));
                    beta.Grad[i] += y.Grad[offsetedIndex] * y.Data[offsetedIndex] * (x.Data[offsetedIndex] - y.Data[offsetedIndex]);
                }
            }
        }
    }
}
