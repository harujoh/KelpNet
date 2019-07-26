using System;

namespace KelpNet
{
    [Serializable]
    public class Softmax<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Softmax";

        public Softmax(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            RealArray<T> y = new T[x.DataLength];

            int indexOffset = 0;

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real<T> maxval = x.Data[indexOffset];

                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < x.Data[indexOffset + i])
                    {
                        maxval = x.Data[indexOffset + i];
                    }
                }

                Real<T> sumval = 0;

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

            return NdArray<T>.Convert(y, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            RealArray<T> gx = new T[y.DataLength];

            int indexOffset = 0;

            for (int b = 0; b < y.BatchCount; b++)
            {
                Real<T> sumdx = 0;

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

            for (int i = 0; i < x.DataLength; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
