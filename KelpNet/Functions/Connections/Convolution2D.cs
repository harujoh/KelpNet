using System;
using System.Drawing;
using Cloo;
using KelpNet.Properties;

namespace KelpNet
{
    [Serializable]
    public class Convolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Convolution2D";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _strideX;
        private readonly int _strideY;
        private readonly int _padX;
        private readonly int _padY;

        public readonly int InputCount;
        public readonly int OutputCount;

        protected override string KernelString
        {
            get
            {
                return Weaver.GetKernelSource(Resources.Convolution2D);
            }
        }

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, name, inputNames, outputNames, gpuEnable)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._strideX = stride;
            this._strideY = stride;
            this._padX = pad;
            this._padY = pad;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, Size stride = new Size(), Size pad = new Size(), bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, name, inputNames, outputNames, gpuEnable)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(1, 1);

            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._strideX = stride.Width;
            this._strideY = stride.Height;
            this._padX = pad.Width;
            this._padY = pad.Height;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(Linear linear) : base(FUNCTION_NAME, linear.Activator, linear.Name, linear.InputNames, linear.OutputNames, linear.GpuEnable)
        {
            this._kWidth = 1;
            this._kHeight = 1;
            this._strideX = 1;
            this._strideY = 1;
            this._padX = 0;
            this._padY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Reshape(OutputCount, InputCount, this._kHeight, this._kWidth);
            this.Bias = linear.Bias;
            this.NoBias = linear.NoBias;
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(OutputCount, InputCount, this._kHeight, this._kWidth);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!NoBias)
            {
                this.Bias = new NdArray(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.ToRealArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            int outputHeight = (int)Math.Floor((x.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((x.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] y = new Real[x.BatchCount * this.OutputCount * outputHeight * outputWidth];

            for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * this.OutputCount * outputHeight * outputWidth;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    int kOchOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    int yChOffset = yBatchOffset + och * outputHeight * outputWidth;

                    for (int oy = 0; oy < outputHeight * this._strideY; oy += this._strideY)
                    {
                        int iyStart = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int iyLimit = this._kHeight + oy - this._padY < x.Shape[1] ? this._kHeight + oy - this._padY : x.Shape[1];

                        for (int ox = 0; ox < outputWidth * this._strideX; ox += this._strideX)
                        {
                            int ixStart = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int ixLimit = this._kWidth + ox - this._padX < x.Shape[2] ? this._kWidth + ox - this._padX : x.Shape[2];

                            int yIndex = yChOffset + oy / this._strideY * outputWidth + ox / this._strideX;

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                int kIchOffset = kOchOffset + ich * this._kHeight * this._kWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = kIchOffset + (iy - oy + this._padY) * this._kWidth + ix - ox + this._padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        y[yIndex] += x.Data[xIndex] * this.Weight.Data[wIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (this.Activator != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            y[resultIndex] = this.Activator.ForwardActivate(y[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (this.Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = this.Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { this.OutputCount, outputHeight, outputWidth }, x.BatchCount, this);
        }

        protected override NdArray NeedPreviousForwardGpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.NoBias ? new Real[OutputCount] : this.Bias.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
            {
                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpub);
                ForwardKernel.SetMemoryArgument(3, gpuY);
                ForwardKernel.SetValueArgument(4, input.Shape[1]);
                ForwardKernel.SetValueArgument(5, input.Shape[2]);
                ForwardKernel.SetValueArgument(6, input.Length);
                ForwardKernel.SetValueArgument(7, outputWidth);
                ForwardKernel.SetValueArgument(8, outputHeight);
                ForwardKernel.SetValueArgument(9, this._strideX);
                ForwardKernel.SetValueArgument(10, this._strideY);
                ForwardKernel.SetValueArgument(11, this._padX);
                ForwardKernel.SetValueArgument(12, this._padY);
                ForwardKernel.SetValueArgument(13, this._kHeight);
                ForwardKernel.SetValueArgument(14, this._kWidth);
                ForwardKernel.SetValueArgument(15, this.OutputCount);
                ForwardKernel.SetValueArgument(16, this.InputCount);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                    null,
                    null
                );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgy = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgy.Length; i++)
            {
                activatedgy[i] = this.Activator.BackwardActivate(y.Grad[i], y.Data[i]);
            }

            return activatedgy;
        }

        void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        this.Bias.Grad[och] += gy[gyIndex];

                        gyIndex++;
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * y.Length;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < y.Shape[0]; och++)
                {
                    int wOchOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = 0; oy < y.Shape[1] * this._strideY; oy += this._strideY)
                    {
                        int iyStart = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int iyLimit = this._kHeight + oy - this._padY < x.Shape[1] ? this._kHeight + oy - this._padY : x.Shape[1];

                        for (int ox = 0; ox < y.Shape[2] * this._strideX; ox += this._strideX)
                        {
                            int ixStart = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int ixLimit = this._kWidth + ox - this._padX < x.Shape[2] ? this._kWidth + ox - this._padX : x.Shape[2];

                            int gyIndex = yBatchOffset + yChOffset + oy / this._strideY * y.Shape[2] + ox / this._strideX;

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                int wIchOffset = wOchOffset + ich * this._kHeight * this._kWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = wIchOffset + (iy - oy + this._padY) * this._kWidth + ix - ox + this._padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        this.Weight.Grad[wIndex] += x.Data[xIndex] * activatedgy[gyIndex];

                                        x.Grad[xIndex] += this.Weight.Data[wIndex] * activatedgy[gyIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            //gyは共通で使用
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                {
                    this.BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    this.BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    this.BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    this.BackwardgWKernel.SetValueArgument(4, this.InputCount);
                    this.BackwardgWKernel.SetValueArgument(5, y.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(6, y.Shape[2]);
                    this.BackwardgWKernel.SetValueArgument(7, y.Length);
                    this.BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                    this.BackwardgWKernel.SetValueArgument(10, x.Length);
                    this.BackwardgWKernel.SetValueArgument(11, this._strideX);
                    this.BackwardgWKernel.SetValueArgument(12, this._strideY);
                    this.BackwardgWKernel.SetValueArgument(13, this._padX);
                    this.BackwardgWKernel.SetValueArgument(14, this._padY);
                    this.BackwardgWKernel.SetValueArgument(15, this._kHeight);
                    this.BackwardgWKernel.SetValueArgument(16, this._kWidth);

                    Weaver.CommandQueue.Execute
                    (
                        this.BackwardgWKernel,
                        null,
                        new long[] { OutputCount * InputCount, this._kHeight, this._kWidth },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
                {
                    this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    this.BackwardgXKernel.SetValueArgument(3, this.InputCount);
                    this.BackwardgXKernel.SetValueArgument(4, y.Shape[0]);
                    this.BackwardgXKernel.SetValueArgument(5, y.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(6, y.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(7, y.Length);
                    this.BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(10, x.Length);
                    this.BackwardgXKernel.SetValueArgument(11, this._strideX);
                    this.BackwardgXKernel.SetValueArgument(12, this._strideY);
                    this.BackwardgXKernel.SetValueArgument(13, this._padX);
                    this.BackwardgXKernel.SetValueArgument(14, this._padY);
                    this.BackwardgXKernel.SetValueArgument(15, this._kHeight);
                    this.BackwardgXKernel.SetValueArgument(16, this._kWidth);

                    Weaver.CommandQueue.Execute
                    (
                        this.BackwardgXKernel,
                        null,
                        new long[] { x.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
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
    }
}
