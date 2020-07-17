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
    public class LRN<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LRN";

        private int n;
        private T k;
        private T alpha;
        private T beta;
        private T[] unitScale;
        private T[] scale;

        public LRN(int n = 5, T? k = null, T? alpha = null, T? beta = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.n = n;
            this.k = k ?? (TVal<T>)2;
            this.alpha = alpha ?? (TVal<T>)1e-4;
            this.beta = beta ?? (TVal<T>)0.75;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case LRN<float> lrnF:
                    lrnF.SingleInputForward = (x) => LRNF.SingleInputForward(x, lrnF.n, out lrnF.unitScale, out lrnF.scale, lrnF.k, lrnF.alpha, lrnF.beta, lrnF);
                    lrnF.SingleOutputBackward = (y, x) => LRNF.SingleOutputBackward(y, x, lrnF.n, lrnF.unitScale, lrnF.scale, lrnF.alpha, lrnF.beta);
                    break;

                case LRN<double> lrnD:
                    lrnD.SingleInputForward = (x) => LRND.SingleInputForward(x, lrnD.n, out lrnD.unitScale, out lrnD.scale, lrnD.k, lrnD.alpha, lrnD.beta, lrnD);
                    lrnD.SingleOutputBackward = (y, x) => LRND.SingleOutputBackward(y, x, lrnD.n, lrnD.unitScale, lrnD.scale, lrnD.alpha, lrnD.beta);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class LRND
#else
    public static class LRNF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, int n, out Real[] unitScale, out Real[] scale, Real k, Real alpha, Real beta, IFunction<Real> lrn)
        {
            int nHalf = n / 2;
            Real[] result = new Real[input.Data.Length];
            Real[] x2 = new Real[input.Data.Length];
            Real[] sumPart = new Real[input.Data.Length];
            unitScale = new Real[input.Data.Length];
            scale = new Real[input.Data.Length];

            for (int i = 0; i < x2.Length; i++)
            {
                x2[i] = input.Data[i] * input.Data[i];
            }

            Array.Copy(x2, sumPart, x2.Length);

            for (int offsetCh = 1; offsetCh < nHalf + 1; offsetCh++)
            {
                int offset = offsetCh * input.Shape[1] * input.Shape[2];

                for (int b = 0; b < input.BatchCount; b++)
                {
                    for (int i = offsetCh; i < input.Shape[0]; i++)
                    {
                        int baseIndex = b * input.Length + i * input.Shape[1] * input.Shape[2];

                        for (int j = 0; j < input.Shape[1] * input.Shape[2]; j++)
                        {
                            sumPart[baseIndex + j] += x2[baseIndex - offset + j];
                            sumPart[baseIndex - offset + j] += x2[baseIndex + j];
                        }
                    }
                }
            }

            //前後nチャンネルで場所の平均を取る
            for (int i = 0; i < sumPart.Length; i++)
            {
                unitScale[i] = k + alpha * sumPart[i];
                scale[i] = Math.Pow(unitScale[i], -beta);
                result[i] = input.Data[i] * scale[i];
            }

            return NdArray.Convert(result, input.Shape, input.BatchCount, lrn);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, int n, Real[] unitScale, Real[] scale, Real alpha, Real beta)
        {
            int nHalf = n / 2;
            Real[] summand = new Real[y.Grad.Length];
            Real[] sumPart = new Real[y.Grad.Length];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                summand[i] = y.Data[i] * y.Grad[i] / unitScale[i];
            }

            Array.Copy(summand, sumPart, summand.Length);

            for (int offsetCh = 1; offsetCh < nHalf + 1; offsetCh++)
            {
                int offset = offsetCh * y.Shape[1] * y.Shape[2];

                for (int b = 0; b < y.BatchCount; b++)
                {
                    for (int i = offsetCh; i < y.Shape[0]; i++)
                    {
                        int baseIndex = b * y.Length + i * y.Shape[1] * y.Shape[2];

                        for (int j = 0; j < y.Shape[1] * y.Shape[2]; j++)
                        {
                            sumPart[baseIndex + j] += summand[baseIndex - offset + j];
                            sumPart[baseIndex - offset + j] += summand[baseIndex + j];
                        }
                    }
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * scale[i] - 2 * alpha * beta * x.Data[i] * sumPart[i];
            }
        }
    }
}

