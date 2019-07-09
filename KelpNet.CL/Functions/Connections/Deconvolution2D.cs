using System;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class Deconvolution2D : KelpNet.CompressibleFunction
    {
        const string FUNCTION_NAME = "Deconvolution2D";

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
                return Weaver.GetKernelSource(Resources.Deconvolution2D);
            }
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, KelpNet.CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, name, inputNames, outputNames, gpuEnable)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._padX = pad;
            this._padY = pad;
            this._strideX = stride;
            this._strideY = stride;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int[] kSize, int[] subSample = null, int[] trim = null, bool noBias = false, Array initialW = null, Array initialb = null, KelpNet.CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, name, inputNames, outputNames, gpuEnable)
        {
            if (subSample == null)
                subSample = new[] { 1, 1 };

            if (trim == null)
                trim = new[] { 0, 0 };

            this._kWidth = kSize[0];
            this._kHeight = kSize[1];
            this._padX = trim[0];
            this._padY = trim[1];
            this.NoBias = noBias;

            this._strideX = subSample[0];
            this._strideY = subSample[1];

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Initialize(initialW, initialb);
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(InputCount, OutputCount, this._kHeight, this._kWidth);
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

        protected override NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * this._strideY + this._kHeight - this._padY * 2;
            int outputWidth = (input.Shape[2] - 1) * this._strideX + this._kWidth - this._padX * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;
            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = this.Weight.Shape[2] * this.Weight.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < this.OutputCount; och++)
                {
                    for (int oy = this._padY; oy < outputHeight + this._padY; oy++)
                    {
                        int iyLimit = oy / this._strideY + 1 < input.Shape[1] ? oy / this._strideY + 1 : input.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this._strideY + 1;

                        for (int ox = this._padX; ox < outputWidth + this._padX; ox++)
                        {
                            int ixLimit = ox / this._strideX + 1 < input.Shape[2] ? ox / this._strideX + 1 : input.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this._strideX + 1;

                            int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._padY) * outputWidth + ox - this._padX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = ich * this.Weight.Shape[1] * kSizeOffset + och * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * this._strideY) * this.Weight.Shape[3] + (ox - ix * this._strideX);

                                        result[outputIndex] += input.Data[inputIndex] * this.Weight.Data[kernelIndex];
                                    }
                                }
                            }

                        }
                    }
                }
            }

            if (this.Activator != null && !NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this._padY; oy < outputHeight + this._padY; oy++)
                        {
                            for (int ox = this._padX; ox < outputWidth + this._padX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._padY) * outputWidth + ox - this._padX;

                                result[outputIndex] += this.Bias.Data[och];
                                result[outputIndex] = this.Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this._padY; oy < outputHeight + this._padY; oy++)
                        {
                            for (int ox = this._padX; ox < outputWidth + this._padX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._padY) * outputWidth + ox - this._padX;

                                result[outputIndex] += this.Bias.Data[och];
                            }
                        }
                    }
                }
            }
            else if (this.Activator != null)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int oy = this._padY; oy < outputHeight + this._padY; oy++)
                        {
                            for (int ox = this._padX; ox < outputWidth + this._padX; ox++)
                            {
                                int outputIndex = batchCount * this.OutputCount * outSizeOffset + och * outSizeOffset + (oy - this._padY) * outputWidth + ox - this._padX;

                                result[outputIndex] = this.Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected override NdArray NeedPreviousForwardGpu(NdArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * this._strideY + this._kHeight - this._padY * 2;
            int outputWidth = (input.Shape[2] - 1) * this._strideX + this._kWidth - this._padX * 2;

            Real[] result = new Real[input.BatchCount * this.OutputCount * outputWidth * outputHeight];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.NoBias ? new Real[OutputCount] : this.Bias.Data))
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
            int gyIndex = 0;

            Real[] activatedgy = new Real[y.Grad.Length];

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    for (int olocation = 0; olocation < y.Shape[1] * y.Shape[2]; olocation++)
                    {
                        activatedgy[gyIndex] = this.Activator.BackwardActivate(y.Grad[gyIndex], y.Data[gyIndex]);
                        gyIndex++;
                    }
                }
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

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    int wOchOffset = och * this.Weight.Shape[2] * this.Weight.Shape[3];
                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = this._padY; oy < y.Shape[1] + this._padY; oy++)
                    {
                        int iyLimit = oy / this._strideY + 1 < x.Shape[1] ? oy / this._strideY + 1 : x.Shape[1];
                        int iyStart = oy - this.Weight.Shape[2] < 0 ? 0 : (oy - this.Weight.Shape[2]) / this._strideY + 1;

                        for (int ox = this._padX; ox < y.Shape[2] + this._padX; ox++)
                        {
                            int ixLimit = ox / this._strideX + 1 < x.Shape[2] ? ox / this._strideX + 1 : x.Shape[2];
                            int ixStart = ox - this.Weight.Shape[3] < 0 ? 0 : (ox - this.Weight.Shape[3]) / this._strideX + 1;

                            int gyIndex = batchCount * y.Length + yChOffset + (oy - this._padY) * y.Shape[2] + ox - this._padX;

                            for (int ich = 0; ich < InputCount; ich++)
                            {
                                int wIchOffset = ich * this.Weight.Shape[1] * this.Weight.Shape[2] * this.Weight.Shape[3] + wOchOffset;
                                int xChOffset = batchCount * x.Length + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;
                                        int wIndex = wIchOffset + (oy - iy * this._strideY) * this.Weight.Shape[3] + (ox - ix * this._strideX);

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
            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, this.Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x.Data))
                {
                    this.BackwardgWKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgWKernel.SetMemoryArgument(1, gpuX);
                    this.BackwardgWKernel.SetMemoryArgument(2, gpugW);
                    this.BackwardgWKernel.SetValueArgument(3, y.BatchCount);
                    this.BackwardgWKernel.SetValueArgument(4, this.OutputCount);
                    this.BackwardgWKernel.SetValueArgument(5, y.Length);
                    this.BackwardgWKernel.SetValueArgument(6, y.Shape[1]);
                    this.BackwardgWKernel.SetValueArgument(7, y.Shape[2]);
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
                        new long[] { this.InputCount * this.OutputCount, this._kHeight, this._kWidth },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, this.Weight.Data))
                {
                    this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                    this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                    this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                    this.BackwardgXKernel.SetValueArgument(3, this.OutputCount);
                    this.BackwardgXKernel.SetValueArgument(4, this.InputCount);
                    this.BackwardgXKernel.SetValueArgument(5, y.Length);
                    this.BackwardgXKernel.SetValueArgument(6, y.Shape[1]);
                    this.BackwardgXKernel.SetValueArgument(7, y.Shape[2]);
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
