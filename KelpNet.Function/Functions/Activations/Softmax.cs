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
    public class Softmax<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Softmax";

        public Softmax(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Softmax<float> softmaxF:
                    softmaxF.SingleInputForward = (x) => SoftmaxF.SingleInputForward(x, softmaxF);
                    softmaxF.SingleOutputBackward = SoftmaxF.SingleOutputBackward;
                    break;

                case Softmax<double> softmaxD:
                    softmaxD.SingleInputForward = (x) => SoftmaxD.SingleInputForward(x, softmaxD);
                    softmaxD.SingleOutputBackward = SoftmaxD.SingleOutputBackward;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SoftmaxD
#else
    public static class SoftmaxF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> softmax)
        {
            Real[] y = new Real[x.Data.Length];

            int indexOffset = 0;

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = x.Data[indexOffset];

                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < x.Data[indexOffset + i])
                    {
                        maxval = x.Data[indexOffset + i];
                    }
                }

                Real sumval = 0;

                for (int i = 0; i < x.Length; i++)
                {
                    y[indexOffset + i] = Math.Exp(x.Data[indexOffset + i] - maxval);
                    sumval += y[indexOffset + i];
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[indexOffset + i] /= sumval;
                }

                indexOffset += x.Length;
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, softmax);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            Real[] gx = new Real[y.Grad.Length];

            int indexOffset = 0;

            for (int b = 0; b < y.BatchCount; b++)
            {
                Real sumdx = 0;

                for (int i = 0; i < y.Length; i++)
                {
                    gx[indexOffset + i] = y.Data[indexOffset + i] * y.Data[indexOffset + i];
                    sumdx += gx[indexOffset + i];
                }

                for (int i = 0; i < y.Length; i++)
                {
                    gx[indexOffset + i] -= y.Data[indexOffset + i] * sumdx;
                }

                indexOffset += y.Length;
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
