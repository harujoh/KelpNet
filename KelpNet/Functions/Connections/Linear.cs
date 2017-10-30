using System;
using System.Collections.Generic;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : CompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "gpuYSum = ForwardActivate(gpuYSum);";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool gpuEnable = false, CompressibleActivation activation = null) : base(name, gpuEnable, FUNCTION_NAME, activation, new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE))
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
                this.Weight.Data = Real.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!noBias)
            {
                this.Bias = new NdArray(outputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(this.Bias.Data, 0, y, i * this.OutputCount, this.Bias.Data.Length);
            }

            return y;
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
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

        protected override NdArray NeedPreviousForwardGpu(NdArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpuY);
                ForwardKernel.SetValueArgument(3, this.OutputCount);
                ForwardKernel.SetValueArgument(4, this.InputCount);

                Weaver.CommandQueue.Execute
                    (
                        ForwardKernel,
                        null,
                        new long[] { OutputCount, x.BatchCount },
                        null,
                        null
                    );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    int index = batchCount * this.OutputCount + i;
                    activatedgY[index] = this.Activator.BackwardActivate(y.Grad[index], y.Data[index]);
                }
            }

            return activatedgY;
        }

        void CalcBiasGrad(Real[] gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    this.Bias.Grad[i] += gy[batchCounter * this.OutputCount + i];
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
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
                        this.Weight.Grad[i * this.InputCount + j] += x.Data[j + batchCount * this.InputCount] * gyData;
                        x.Grad[j + batchCount * this.InputCount] += this.Weight.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                {
                    BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgWKernel.SetValueArgument(4, this.OutputCount);
                    BackwardgWKernel.SetValueArgument(5, this.InputCount);

                    Weaver.CommandQueue.Execute
                    (
                        BackwardgWKernel,
                        null,
                        new long[] { this.InputCount, this.OutputCount },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
                {
                    BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    BackwardgXKernel.SetValueArgument(3, y.BatchCount);
                    BackwardgXKernel.SetValueArgument(4, this.OutputCount);
                    BackwardgXKernel.SetValueArgument(5, this.InputCount);

                    Weaver.CommandQueue.Execute
                    (
                        BackwardgXKernel,
                        null,
                        new long[] { this.InputCount, y.BatchCount },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
