using System;

namespace KelpNet
{
    [Serializable]
    public class Swish : SingleInputFunction
    {
        const string FUNCTION_NAME = "Swish";

        public NdArray Beta;

        public Swish(int[] betaShape, double beta = 1.0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Beta = new NdArray(betaShape);
            this.Beta.Fill(beta);

            this.Parameters = new[] { this.Beta };
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    int offsetedIndex = b * x.Length + i;
                    result[offsetedIndex] = x.Data[offsetedIndex] * (Math.Tanh(x.Data[offsetedIndex] * Beta.Data[i] * 0.5) * 0.5 + 0.5);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    int offsetedIndex = b * x.Length + i;
                    Real sig = Math.Tanh(Beta.Data[i] * x.Data[offsetedIndex] * 0.5) * 0.5 + 0.5;
                    Real by = Beta.Data[i] * x.Data[offsetedIndex] * sig;

                    x.Grad[offsetedIndex] += y.Grad[offsetedIndex] * (by + sig * (1 - by));
                    Beta.Grad[i] += y.Grad[offsetedIndex] * y.Data[offsetedIndex] * (x.Data[offsetedIndex] - y.Data[offsetedIndex]);
                }
            }
        }
    }
}
