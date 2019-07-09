using System;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class Convolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Convolution2D";

        public NdArray Weight;
        public NdArray Bias;

        public bool NoBias;

        public int KernelWidth;
        public int KernelHeight;
        public int StrideX;
        public int StrideY;
        public int PadX;
        public int PadY;

        public int InputCount;
        public int OutputCount;

        public Convolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Convolution2D), activation, name, inputNames, outputNames, gpuEnable)
        {
            this.KernelWidth = kernelSize;
            this.KernelHeight = kernelSize;
            this.StrideX = stride;
            this.StrideY = stride;
            this.PadX = pad;
            this.PadY = pad;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Convolution2D), activation, name, inputNames, outputNames, gpuEnable)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this.KernelWidth = kernelSize[0];
            this.KernelHeight = kernelSize[1];
            this.StrideX = stride[0];
            this.StrideY = stride[1];
            this.PadX = pad[0];
            this.PadY = pad[1];
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(Linear linear) : base(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Convolution2D),linear.Activator, linear.Name, linear.InputNames, linear.OutputNames, linear.IsParallel)
        {
            this.KernelWidth = 1;
            this.KernelHeight = 1;
            this.StrideX = 1;
            this.StrideY = 1;
            this.PadX = 0;
            this.PadY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Reshape(OutputCount, InputCount, this.KernelHeight, this.KernelWidth);
            this.Bias = linear.Bias;
            this.NoBias = linear.NoBias;
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(OutputCount, InputCount, this.KernelHeight, this.KernelWidth);
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
            int outputHeight = (int)Math.Floor((x.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = (int)Math.Floor((x.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;

            Real[] y = new Real[x.BatchCount * this.OutputCount * outputHeight * outputWidth];

            for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * this.OutputCount * outputHeight * outputWidth;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    int kOchOffset = och * this.InputCount * this.KernelHeight * this.KernelWidth;

                    int yChOffset = yBatchOffset + och * outputHeight * outputWidth;

                    for (int oy = 0; oy < outputHeight * this.StrideY; oy += this.StrideY)
                    {
                        int iyStart = oy - this.PadY < 0 ? 0 : oy - this.PadY;
                        int iyLimit = this.KernelHeight + oy - this.PadY < x.Shape[1] ? this.KernelHeight + oy - this.PadY : x.Shape[1];

                        for (int ox = 0; ox < outputWidth * this.StrideX; ox += this.StrideX)
                        {
                            int ixStart = ox - this.PadX < 0 ? 0 : ox - this.PadX;
                            int ixLimit = this.KernelWidth + ox - this.PadX < x.Shape[2] ? this.KernelWidth + ox - this.PadX : x.Shape[2];

                            int yIndex = yChOffset + oy / this.StrideY * outputWidth + ox / this.StrideX;

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                int kIchOffset = kOchOffset + ich * this.KernelHeight * this.KernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = kIchOffset + (iy - oy + this.PadY) * this.KernelWidth + ix - ox + this.PadX;
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
            int outputHeight = (int)Math.Floor((input.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.NoBias ? new Real[OutputCount] : this.Bias.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
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
                ForwardKernel.SetValueArgument(9, this.StrideX);
                ForwardKernel.SetValueArgument(10, this.StrideY);
                ForwardKernel.SetValueArgument(11, this.PadX);
                ForwardKernel.SetValueArgument(12, this.PadY);
                ForwardKernel.SetValueArgument(13, this.KernelHeight);
                ForwardKernel.SetValueArgument(14, this.KernelWidth);
                ForwardKernel.SetValueArgument(15, this.OutputCount);
                ForwardKernel.SetValueArgument(16, this.InputCount);

                OpenCL.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                    null,
                    null
                );

                OpenCL.CommandQueue.Finish();
                OpenCL.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
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
                    int wOchOffset = och * this.InputCount * this.KernelHeight * this.KernelWidth;

                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = 0; oy < y.Shape[1] * this.StrideY; oy += this.StrideY)
                    {
                        int iyStart = oy - this.PadY < 0 ? 0 : oy - this.PadY;
                        int iyLimit = this.KernelHeight + oy - this.PadY < x.Shape[1] ? this.KernelHeight + oy - this.PadY : x.Shape[1];

                        for (int ox = 0; ox < y.Shape[2] * this.StrideX; ox += this.StrideX)
                        {
                            int ixStart = ox - this.PadX < 0 ? 0 : ox - this.PadX;
                            int ixLimit = this.KernelWidth + ox - this.PadX < x.Shape[2] ? this.KernelWidth + ox - this.PadX : x.Shape[2];

                            int gyIndex = yBatchOffset + yChOffset + oy / this.StrideY * y.Shape[2] + ox / this.StrideX;

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                int wIchOffset = wOchOffset + ich * this.KernelHeight * this.KernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = wIchOffset + (iy - oy + this.PadY) * this.KernelWidth + ix - ox + this.PadX;
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

            int kyStartPrevOffset = KernelHeight - PadY - x.Shape[1];
            int kxStartPrevOffset = KernelWidth - PadX - x.Shape[2];

            //gyは共通で使用
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
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
                    this.BackwardgWKernel.SetValueArgument(11, this.StrideX);
                    this.BackwardgWKernel.SetValueArgument(12, this.StrideY);
                    this.BackwardgWKernel.SetValueArgument(13, this.PadX);
                    this.BackwardgWKernel.SetValueArgument(14, this.PadY);
                    this.BackwardgWKernel.SetValueArgument(15, this.KernelHeight);
                    this.BackwardgWKernel.SetValueArgument(16, this.KernelWidth);

                    OpenCL.CommandQueue.Execute
                    (
                        this.BackwardgWKernel,
                        null,
                        new long[] { OutputCount * InputCount, this.KernelHeight, this.KernelWidth },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(OpenCL.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
                {
                    this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    this.BackwardgXKernel.SetValueArgument(3, y.Length);
                    this.BackwardgXKernel.SetValueArgument(4, y.Shape[0]);
                    this.BackwardgXKernel.SetValueArgument(5, y.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(6, y.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(7, x.Length);
                    this.BackwardgXKernel.SetValueArgument(8, x.Shape[0]);
                    this.BackwardgXKernel.SetValueArgument(9, x.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(10, x.Shape[2]);
                    this.BackwardgXKernel.SetValueArgument(11, this.StrideX);
                    this.BackwardgXKernel.SetValueArgument(12, this.StrideY);
                    this.BackwardgXKernel.SetValueArgument(13, this.PadX);
                    this.BackwardgXKernel.SetValueArgument(14, this.PadY);
                    this.BackwardgXKernel.SetValueArgument(15, this.KernelWidth);
                    this.BackwardgXKernel.SetValueArgument(16, this.KernelHeight);
                    this.BackwardgXKernel.SetValueArgument(17, kxStartPrevOffset);
                    this.BackwardgXKernel.SetValueArgument(18, kyStartPrevOffset);

                    OpenCL.CommandQueue.Execute
                    (
                        this.BackwardgXKernel,
                        null,
                        new long[] { x.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
                        null,
                        null
                    );

                    OpenCL.CommandQueue.Finish();
                    OpenCL.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
