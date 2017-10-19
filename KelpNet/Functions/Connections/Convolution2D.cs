using System;
using System.Collections.Generic;
using System.Drawing;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Convolution2D";
        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "localResult = ForwardActivate(localResult);";

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

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool gpuEnable = false, CompressibleActivation activation = null) : base(name, gpuEnable, FUNCTION_NAME, activation, new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE))
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

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, Size stride = new Size(), Size pad = new Size(), bool noBias = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, bool gpuEnable = false, CompressibleActivation activation = null) : base(name, gpuEnable, FUNCTION_NAME, activation, new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE))
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

        public Convolution2D(Linear linear) : base(linear.Name, linear.GpuEnable, FUNCTION_NAME, linear.Activation, new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE))
        {
            this._kWidth = 1;
            this._kHeight = 1;
            this._strideX = 1;
            this._strideY = 1;
            this._padX = 0;
            this._padY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Shape = new[] { OutputCount, InputCount, this._kHeight, this._kWidth };
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
                this.Weight.Data = Real.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!NoBias)
            {
                this.Bias = new NdArray(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
            {
                int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    //Wインデックス用
                    int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    for (int oy = 0; oy < outputHeight * this._strideY; oy += this._strideY)
                    {
                        int kyStartIndex = oy - this._padY < 0 ? 0 : oy - this._padY;
                        int kyLimit = this._kHeight + oy - this._padY < input.Shape[1] ? this._kHeight + oy - this._padY : input.Shape[1];

                        for (int ox = 0; ox < outputWidth * this._strideX; ox += this._strideX)
                        {
                            int kxStartIndex = ox - this._padX < 0 ? 0 : ox - this._padX;
                            int kxLimit = this._kWidth + ox - this._padX < input.Shape[2] ? this._kWidth + ox - this._padX : input.Shape[2];

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                //Wインデックス用
                                int inChOffset = ich * this._kHeight * this._kWidth;

                                //inputインデックス用
                                int inputOffset = ich * input.Shape[1] * input.Shape[2];

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        int wIndex = outChOffset + inChOffset + (ky - oy + this._padY) * this._kWidth + kx - ox + this._padX;
                                        int inputIndex = inputOffset + ky * input.Shape[2] + kx + batchCounter * input.Length;

                                        result[resultIndex] += input.Data[inputIndex] * this.Weight.Data[wIndex];
                                    }
                                }
                            }

                            resultIndex++;
                        }
                    }
                }
            }

            if (this.Activation != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += this.Bias.Data[och];
                            result[resultIndex] = this.Activation.ForwardActivate(result[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] += this.Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (this.Activation != null)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            result[resultIndex] = this.Activation.ForwardActivate(result[resultIndex]);
                            resultIndex++;
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected override NdArray NeedPreviousForwardGpu(NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._strideX) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
            using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
            using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Bias.Data))
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
                        activatedgy[gyIndex] = this.Activation.BackwardActivate(y.Grad[gyIndex], y.Data[gyIndex]);
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
            Real[] activatedgy = this.Activation != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    //gWインデックス用
                    int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                    for (int oy = 0; oy < y.Shape[1] * this._strideY; oy += this._strideY)
                    {
                        //計算省略のためにジャンプ
                        int kyStartIndex = this._padY - oy < 0 ? 0 : this._padY - oy;
                        int kyLimit = this._kHeight < x.Shape[1] - oy + this._padY ? this._kHeight : x.Shape[1] - oy + this._padY;

                        for (int ox = 0; ox < y.Shape[2] * this._strideX; ox += this._strideX)
                        {
                            //計算省略のためにジャンプ
                            int kxStartIndex = this._padX - ox < 0 ? 0 : this._padX - ox;
                            int kxLimit = this._kWidth < x.Shape[2] - ox + this._padX ? this._kWidth : x.Shape[2] - ox + this._padX;

                            int gyIndex = batchCounter * y.Length + och * y.Shape[1] * y.Shape[2] + oy * y.Shape[2] + ox;

                            Real gyData = activatedgy[gyIndex];

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                //gWインデックス用
                                int inChOffset = ich * this._kHeight * this._kWidth;

                                //inputインデックス用
                                int inputOffset = ich * x.Shape[1] * x.Shape[2] + batchCounter * x.Length;

                                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                                {
                                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                                    {
                                        //WとgWのshapeは等しい
                                        int wIndex = outChOffset + inChOffset + ky * this._kWidth + kx;

                                        //xとgxのshapeは等しい
                                        int inputIndex = inputOffset + (ky + oy - this._padY) * x.Shape[2] + kx + ox - this._padX;

                                        this.Weight.Grad[wIndex] += x.Data[inputIndex] * gyData;

                                        x.Grad[inputIndex] += this.Weight.Data[wIndex] * gyData;
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
            Real[] activatedgy = this.Activation != null ? GetActivatedgy(y) : y.Grad;
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
                    this.BackwardgWKernel.SetValueArgument(5, y.Shape[0]);
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
                    this.BackwardgXKernel.SetValueArgument(3, this.OutputCount);
                    this.BackwardgXKernel.SetValueArgument(4, this.InputCount);
                    this.BackwardgXKernel.SetValueArgument(5, y.Shape[0]);
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
                        new long[] { y.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
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
