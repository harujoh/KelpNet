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
        private RealArray<T> unitScale;
        private RealArray<T> scale;

        public LRN(int n = 5, double k = 2, double alpha = 1e-4, double beta = 0.75, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.n = n;
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        private unsafe NdArray<T> NeedPreviousForwardCpu(NdArray<T> input)
        {
            int nHalf = n / 2;
            RealArray<T> result = new T[input.DataLength];
            RealArray<T> x2 = new T[input.DataLength];
            RealArray<T> sumPart = new T[input.DataLength];
            unitScale = new T[input.DataLength];
            scale = new T[input.DataLength];

            for (int i = 0; i < input.DataLength; i++)
            {
                x2[i] = input.Data[i] * input.Data[i];
            }
            //Array.Copy(x2, sumPart, input.DataLength);
            Buffer.MemoryCopy((byte*)x2.Ptr, (byte*)sumPart.Ptr, input.DataLength * sizeof(T), input.DataLength * sizeof(T));

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
            for (int i = 0; i < input.DataLength; i++)
            {
                this.unitScale[i] = this.k + this.alpha * sumPart[i];
                this.scale[i] = (Real<T>)Math.Pow(this.unitScale[i], -this.beta);
                result[i] *= this.scale[i];
            }

            return NdArray<T>.Convert(result, input.Shape, input.BatchCount, this);
        }

        private unsafe void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            int nHalf = n / 2;
            RealArray<T> summand = new T[y.DataLength];
            RealArray<T> sumPart = new T[y.DataLength];

            for (int i = 0; i < y.DataLength; i++)
            {
                summand[i] = y.Data[i] * y.Grad[i] / this.unitScale[i];
            }

            //Array.Copy(summand, sumPart, y.DataLength);
            Buffer.MemoryCopy((byte*)summand.Ptr, (byte*)sumPart.Ptr, y.DataLength * sizeof(T), y.DataLength * sizeof(T));

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

            for (int i = 0; i < x.DataLength; i++)
            {
                x.Grad[i] += y.Grad[i] * this.scale[i] - 2 * this.alpha * this.beta * y.Data[i] * sumPart[i];
            }
        }
    }
}
