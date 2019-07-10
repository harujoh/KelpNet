using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class Linear : CompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        public NdArray Weight;
        public NdArray Bias;

        public bool NoBias;

        public int InputCount;
        public int OutputCount;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(activation, name, inputNames, outputNames)
        {
            this.OutputCount = outputCount;
            this.InputCount = inputCount;

            this.Weight = new NdArray(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!noBias)
            {
                this.Bias = new NdArray(outputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.ToRealArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        protected Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(this.Bias.Data, 0, y, i * this.OutputCount, this.Bias.Data.Length);
            }

            return y;
        }

        public override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    for (int j = 0; j < this.InputCount; j++)
                    {
                        y[batchCount * this.OutputCount + i] += x.Data[batchCount * this.InputCount + j] * this.Weight.Data[i * this.InputCount + j];
                    }
                }
            }

            if (this.Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = this.Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgY.Length; i++)
            {
                activatedgY[i] = this.Activator.BackwardActivate(y.Grad[i], y.Data[i]);
            }

            return activatedgY;
        }

        protected void CalcBiasGrad(Real[] gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    this.Bias.Grad[i] += gy[batchCounter * this.OutputCount + i];
                }
            }
        }

        public override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * this.OutputCount];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.Weight.Grad[i * this.InputCount + j] += x.Data[batchCount * this.InputCount + j] * gyData;
                        x.Grad[batchCount * this.InputCount + j] += this.Weight.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
