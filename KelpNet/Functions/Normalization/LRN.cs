using System;

namespace KelpNet
{
    public class LRN : SingleInputFunction
    {
        const string FUNCTION_NAME = "LRN";

        private int n;
        private Real k;
        private Real alpha;
        private Real beta;
        private Real[] unitScale;
        private Real[] scale;

        public LRN(int n = 5, double k = 2, double alpha = 1e-4, double beta = 0.75, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.n = n;
            this.k = (Real)k;
            this.alpha = (Real)alpha;
            this.beta = (Real)beta;
        }

        public override NdArray SingleInputForward(NdArray input)
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
                this.unitScale[i] = this.k + this.alpha * sumPart[i];
                this.scale[i] = Math.Pow(this.unitScale[i], -this.beta);
                result[i] = input.Data[i] * this.scale[i];
            }

            return NdArray.Convert(result, input.Shape, input.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            int nHalf = n / 2;
            Real[] summand = new Real[y.Grad.Length];
            Real[] sumPart = new Real[y.Grad.Length];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                summand[i] = y.Data[i] * y.Grad[i] / this.unitScale[i];
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
                x.Grad[i] += y.Grad[i] * this.scale[i] - 2 * this.alpha * this.beta * x.Data[i] * sumPart[i];
            }
        }
    }
}
