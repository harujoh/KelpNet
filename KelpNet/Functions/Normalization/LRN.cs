using System;

namespace KelpNet
{
    public class LRN<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LRN";

        private int n;
        private Real<T> k;
        private Real<T> alpha;
        private Real<T> beta;
        private Real<T>[] unitScale;
        private Real<T>[] scale;

        public LRN(int n = 5, double k = 2, double alpha = 1e-4, double beta = 0.75, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.n = n;
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        private NdArray<T> NeedPreviousForwardCpu(NdArray<T> input)
        {
            int nHalf = n / 2;
            Real<T>[] result = new Real<T>[input.Data.Length];
            Real<T>[] x2 = new Real<T>[input.Data.Length];
            Real<T>[] sumPart = new Real<T>[input.Data.Length];
            unitScale = new Real<T>[input.Data.Length];
            scale = new Real<T>[input.Data.Length];

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
                this.scale[i] = (Real<T>)Math.Pow(this.unitScale[i], -this.beta);
                result[i] *= this.scale[i];
            }

            return NdArray<T>.Convert(result, input.Shape, input.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            int nHalf = n / 2;
            Real<T>[] summand = new Real<T>[y.Grad.Length];
            Real<T>[] sumPart = new Real<T>[y.Grad.Length];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                summand[i] = y.Data[i] * y.Grad[i] / this.unitScale[i];
            }

            Array.Copy(summand, sumPart, summand.Length);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int ich = 0; ich < y.Shape[0]; ich++)
                {
                    for (int location = 0; location < y.Shape[1] * y.Shape[2]; location++)
                    {
                        int baseIndex = b * y.Length + ich * y.Shape[1] * y.Shape[2] + location;

                        for (int offsetCh = 1; offsetCh < nHalf; offsetCh++)
                        {
                            if (ich - offsetCh > 0)
                            {
                                int offsetIndex = b * y.Length + (ich - offsetCh) * y.Shape[1] * y.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }

                            if (ich + offsetCh < y.Shape[0])
                            {
                                int offsetIndex = b * y.Length + (ich + offsetCh) * y.Shape[1] * y.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * this.scale[i] - 2 * this.alpha * this.beta * y.Data[i] * sumPart[i];
            }
        }
    }
}
