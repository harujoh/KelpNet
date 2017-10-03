using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Normalization
{
    public class LRN : NeedPreviousDataFunction
    {
        const string FUNCTION_NAME = "LRN";

        private int n;
        private Real k;
        private Real alpha;
        private Real beta;
        private Real[] unitScale;
        private Real[] scale;

        public LRN(int n = 5, double k = 2, double alpha = 1e-4, double beta = 0.75, string name = FUNCTION_NAME) : base(name)
        {
            this.n = n;
            this.k = (Real)k;
            this.alpha = (Real)alpha;
            this.beta = (Real)beta;

            NeedPreviousForward = NeedPreviousForwardCpu;
            NeedPreviousBackward = NeedPreviousBackwardCpu;
        }

        public BatchArray NeedPreviousForwardCpu(BatchArray input)
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


            for (int b = 0; b < input.BatchCount; b++)
            {
                for (int ich = 0; ich < input.Shape[0]; ich++)
                {
                    for (int location = 0; location < input.Shape[1] * input.Shape[2]; location++)
                    {
                        int baseIndex = b * input.Length + ich * input.Shape[1] * input.Shape[2] + location;

                        for (int offsetCh = 1; offsetCh < nHalf; offsetCh++)
                        {
                            if (ich - offsetCh > 0)
                            {
                                int offsetIndex = b * input.Length + (ich - offsetCh) * input.Shape[1] * input.Shape[2] + location;
                                sumPart[baseIndex] += x2[offsetIndex];
                            }

                            if (ich + offsetCh < input.Shape[0])
                            {
                                int offsetIndex = b * input.Length + (ich + offsetCh) * input.Shape[1] * input.Shape[2] + location;
                                sumPart[baseIndex] += x2[offsetIndex];
                            }
                        }
                    }
                }
            }

            //前後nチャンネルで場所の平均を取る
            for (int i = 0; i < sumPart.Length; i++)
            {
                this.unitScale[i] = this.k + this.alpha * sumPart[i];
                this.scale[i] = Math.Pow(this.unitScale[i], -this.beta);
                result[i] *= this.scale[i];
            }

            return BatchArray.Convert(result, input.Shape, input.BatchCount);
        }

        public BatchArray NeedPreviousBackwardCpu(BatchArray gy, BatchArray prevInput, BatchArray prevOutput)
        {
            int nHalf = n / 2;
            Real[] gx = new Real[gy.Data.Length];
            Real[] summand = new Real[gy.Data.Length];
            Real[] sumPart = new Real[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                summand[i] = prevOutput.Data[i] * gy.Data[i] / this.unitScale[i];
            }

            Array.Copy(summand, sumPart, summand.Length);

            for (int b = 0; b < gy.BatchCount; b++)
            {
                for (int ich = 0; ich < gy.Shape[0]; ich++)
                {
                    for (int location = 0; location < gy.Shape[1] * gy.Shape[2]; location++)
                    {
                        int baseIndex = b * gy.Length + ich * gy.Shape[1] * gy.Shape[2] + location;

                        for (int offsetCh = 1; offsetCh < nHalf; offsetCh++)
                        {
                            if (ich - offsetCh > 0)
                            {
                                int offsetIndex = b * gy.Length + (ich - offsetCh) * gy.Shape[1] * gy.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }

                            if (ich + offsetCh < gy.Shape[0])
                            {
                                int offsetIndex = b * gy.Length + (ich + offsetCh) * gy.Shape[1] * gy.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = gy.Data[i] * this.scale[i] - 2 * this.alpha * this.beta * prevInput.Data[i] * sumPart[i];
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
