using System;

namespace KelpNet
{
    [Serializable]
    public class Softmax : SingleInputFunction
    {
        private const string FUNCTION_NAME = "Softmax";

        public Softmax(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray SingleInputForward(NdArray x)
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

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
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
