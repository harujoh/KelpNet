using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Activations;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousInputFunction
    {
        private readonly Activation _activation;
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _stride;
        private readonly int _padX;
        private readonly int _padY;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Conv2D", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputChannels, outputChannels)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._stride = stride;
            this._padX = pad;
            this._padY = pad;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this._activation = activation ?? new DummyActivation();
            if (IsGpu)
            {
                this.ForwardKernelSource = this._activation.ForwardActivateFunctionString + ForwardKernelSource;
                this.BackwardKernelSource = this._activation.BackwardActivateFunctionString + BackwardKernelSource;
            }

            this.Initialize(initialW, initialb, isGpu);
        }

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, int stride = 1, Size pad = new Size(), bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Conv2D", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputChannels, outputChannels)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._stride = stride;
            this._padX = pad.Width;
            this._padY = pad.Height;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this._activation = activation ?? new DummyActivation();
            if (IsGpu)
            {
                this.ForwardKernelSource = this._activation.ForwardActivateFunctionString + ForwardKernelSource;
                this.BackwardKernelSource = this._activation.BackwardActivateFunctionString + BackwardKernelSource;
            }

            this.Initialize(initialW, initialb, isGpu);
        }

        void Initialize(Real[,,,] initialW = null, Real[] initialb = null, bool isGpu = true)
        {
            this.W = new NdArray(OutputCount, InputCount, this._kHeight, this._kWidth);
            this.gW = NdArray.ZerosLike(this.W);

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                this.W.Data = initialW.Cast<Real>().ToArray();
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = new NdArray(OutputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (this.Parameters.Length > 1)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, "Convolution2DForward");
                BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, "Convolution2DBackward");
            }
        }

        public override string ForwardKernelSource { get; } =
@"
__kernel void Convolution2DForward(
    const __global __read_only  Real* gpuX,
    const __global __read_only  Real* gpuW,
    const __global __read_only  Real* gpub,
          __global __write_only Real* gpuY,
    const int inputShape1,
    const int inputShape2,
    const int inputLength,
    const int outputWidth,
    const int outputHeight,
    const int stride,
    const int padX,
    const int padY,
    const int kHeight,
    const int kWidth,
    const int OutputCount,
    const int InputCount)
{
    int batchCounter = get_global_id(0) / OutputCount;
    int och = get_global_id(0) % OutputCount;
    int oy = get_global_id(1) * stride - padY;
    int ox = get_global_id(2) * stride - padX;

    Real localResult = 0;

    gpuW += och* InputCount * kHeight* kWidth;
    gpuX += batchCounter * inputLength;

    int kyStartIndex = oy < 0 ? 0 : oy;
    int kyLimit = kHeight + oy < inputShape1 ? kHeight + oy : inputShape1;

    int kxStartIndex = ox < 0 ? 0 : ox;
    int kxLimit = kWidth + ox < inputShape2 ? kWidth + ox : inputShape2;

    for (int ich = 0; ich < InputCount; ich++)
    {
        for (int ky = kyStartIndex; ky < kyLimit; ky++)
        {
            for (int kx = kxStartIndex; kx < kxLimit; kx++)
            {
                int inputIndex = ich * inputShape1 * inputShape2 + ky * inputShape2 + kx;
                int wIndex = ich * kHeight * kWidth + (ky - oy) * kWidth + kx - ox;

                localResult += gpuX[inputIndex] * gpuW[wIndex];
            }
        }
    }

    int index = batchCounter * OutputCount * outputHeight * outputWidth + och * outputHeight * outputWidth + get_global_id(1) * outputWidth + get_global_id(2);
    gpuY[index] = localResult + gpub[och];
    ForwardActivate(gpuY + index);
}";

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._stride) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._stride) + 1;

            Real[] result = new Real[this.OutputCount * outputHeight * outputWidth * input.BatchCount];

            if (!IsGpu)
            {
                for (int batchCounter = 0; batchCounter < input.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        //Wインデックス用
                        int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                        for (int oy = 0; oy < outputHeight * this._stride; oy += this._stride)
                        {
                            int kyStartIndex = oy - this._padY < 0 ? 0 : oy - this._padY;
                            int kyLimit = this._kHeight + oy - this._padY < input.Shape[1] ? this._kHeight + oy - this._padY : input.Shape[1];

                            for (int ox = 0; ox < outputWidth * this._stride; ox += this._stride)
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

                                            result[resultIndex] += input.Data[inputIndex] * this.W.Data[wIndex];
                                        }
                                    }
                                }

                                result[resultIndex] += this.b.Data[och];
                                this._activation.ForwardActivate(ref result[resultIndex]);
                                resultIndex++;
                            }
                        }
                    }
                }
            }
            else
            {
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpub = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.b.Data))
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
                    ForwardKernel.SetValueArgument(9, this._stride);
                    ForwardKernel.SetValueArgument(10, this._padX);
                    ForwardKernel.SetValueArgument(11, this._padY);
                    ForwardKernel.SetValueArgument(12, this._kHeight);
                    ForwardKernel.SetValueArgument(13, this._kWidth);
                    ForwardKernel.SetValueArgument(14, this.OutputCount);
                    ForwardKernel.SetValueArgument(15, this.InputCount);

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
            }

            BatchArray output = BatchArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount);
            if (!(this._activation is DummyActivation))
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        public override string BackwardKernelSource { get; } =
@"
__kernel void Convolution2DBackward(
	const __global __read_only  Real* gpugY,
	const __global __read_only  Real* gpuY,
	const __global __read_only  Real* gpuX,
	const __global __read_only  Real* gpuW, 
	      __global __read_write Real* gpugX, 
	      __global __read_write Real* tmpgW, 
	const int outputCount,
	const int inputCount,
    const int gyShape0,
    const int gyShape1,
    const int gyShape2,
    const int xShape1,
    const int xShape2,
    const int xLength,
    const int stride,
	const int padX,
	const int padY,
	const int kHeight,
	const int kWidth)
{
    int batchCounter = get_global_id(0);
	int ich = get_global_id(1);

    int wLength = outputCount * inputCount * kHeight * kWidth;

    gpuW += ich * kHeight * kWidth;
    tmpgW += batchCounter * wLength + ich * kHeight * kWidth;

    gpuX += batchCounter * xLength + ich * xShape1 * xShape2;
    gpugX += batchCounter * xLength + ich * xShape1 * xShape2; 

    gpugY += batchCounter * gyShape0 * gyShape1 * gyShape2;
    gpuY += batchCounter * gyShape0 * gyShape1 * gyShape2;

    int index = 0;

    for (int och = 0; och < gyShape0; och++)
    {
        int wIndex = och * inputCount * kHeight * kWidth;

        for (int oy = 0; oy < gyShape1 * stride; oy += stride)
        {
            int kyStartIndex = padY < oy ? 0 : padY - oy;
            int kyLimit = kHeight < xShape1 - oy + padY ? kHeight : xShape1 - oy + padY;

            for (int ox = 0; ox < gyShape2 * stride; ox += stride)
            {
                int kxStartIndex = padX < ox ? 0 : padX - ox;
                int kxLimit = kWidth < xShape2 - ox + padX ? kWidth : xShape2 - ox + padX;

                Real gyData = gpugY[index];
                BackwardActivate(gpuY[index], &gyData);
                index++;

                for (int ky = kyStartIndex; ky < kyLimit; ky++)
                {
                    for (int kx = kxStartIndex; kx < kxLimit; kx++)
                    {
                        int inputIndex = (ky + oy - padY) * xShape2 + kx + ox - padX;

                        tmpgW[wIndex + ky * kWidth + kx] += gpuX[inputIndex] * gyData;
                        gpugX[inputIndex] += gpuW[wIndex + ky * kWidth + kx] * gyData;
                    }
                }
            }
        }
    }
}";

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray x)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (!(this._activation is DummyActivation))
            {
                prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;
                this._prevOutput.RemoveAt(this._prevOutput.Count - 1);
            }

            Real[] gx = new Real[x.Data.Length];

            if (!IsGpu)
            {
                for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
                {
                    for (int och = 0; och < gy.Shape[0]; och++)
                    {
                        //gWインデックス用
                        int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                        for (int oy = 0; oy < gy.Shape[1] * this._stride; oy += this._stride)
                        {
                            //計算省略のためにジャンプ
                            int kyStartIndex = this._padY - oy < 0 ? 0 : this._padY - oy;
                            int kyLimit = this._kHeight < x.Shape[1] - oy + this._padY ? this._kHeight : x.Shape[1] - oy + this._padY;

                            for (int ox = 0; ox < gy.Shape[2] * this._stride; ox += this._stride)
                            {
                                //計算省略のためにジャンプ
                                int kxStartIndex = this._padX - ox < 0 ? 0 : this._padX - ox;
                                int kxLimit = this._kWidth < x.Shape[2] - ox + this._padX ? this._kWidth : x.Shape[2] - ox + this._padX;

                                int gyIndex = batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[1] + ox;

                                Real gyData = gy.Data[gyIndex]; //gyIndex = ch * ox * oy
                                this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);

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

                                            this.gW.Data[wIndex] += x.Data[inputIndex] * gyData;

                                            gx[inputIndex] += this.W.Data[wIndex] * gyData;
                                        }
                                    }
                                }

                                this.gb.Data[och] += gyData;
                            }
                        }
                    }
                }
            }
            else
            {
                //集計用
                Real[] tmpgWData = new Real[gy.BatchCount * this.gW.Data.Length];
                
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, gy.Data))
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutputData))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gx))
                using (ComputeBuffer<Real> tmpgW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.AllocateHostPointer, this.gW.Data.Length * gy.BatchCount))
                {
                    BackwardKernel.SetMemoryArgument(0, gpugY);
                    BackwardKernel.SetMemoryArgument(1, gpuY);
                    BackwardKernel.SetMemoryArgument(2, gpuX);
                    BackwardKernel.SetMemoryArgument(3, gpuW);
                    BackwardKernel.SetMemoryArgument(4, gpugX);
                    BackwardKernel.SetMemoryArgument(5, tmpgW);
                    BackwardKernel.SetValueArgument(6, this.OutputCount);
                    BackwardKernel.SetValueArgument(7, this.InputCount);
                    BackwardKernel.SetValueArgument(8, gy.Shape[0]);
                    BackwardKernel.SetValueArgument(9, gy.Shape[1]);
                    BackwardKernel.SetValueArgument(10, gy.Shape[2]);
                    BackwardKernel.SetValueArgument(11, x.Shape[1]);
                    BackwardKernel.SetValueArgument(12, x.Shape[2]);
                    BackwardKernel.SetValueArgument(13, x.Length);
                    BackwardKernel.SetValueArgument(14, this._stride);
                    BackwardKernel.SetValueArgument(15, this._padX);
                    BackwardKernel.SetValueArgument(16, this._padY);
                    BackwardKernel.SetValueArgument(17, this._kHeight);
                    BackwardKernel.SetValueArgument(18, this._kWidth);

                    Weaver.CommandQueue.Execute
                    (
                        BackwardKernel,
                        null,
                        new long[] { gy.BatchCount, x.Shape[0] },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(tmpgW, ref tmpgWData, true, null);

                    //集計処理
                    int gyIndex = 0;
                    for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
                    {
                        for (int i = 0; i < this.gW.Data.Length; i++)
                        {
                            this.gW.Data[i] += tmpgWData[batchCounter * this.gW.Data.Length + i];
                        }

                        for (int och = 0; och < gy.Shape[0]; och++)
                        {
                            for (int oy = 0; oy < gy.Shape[1]; oy++)
                            {
                                for (int ox = 0; ox < gy.Shape[2]; ox++)
                                {
                                    Real gyData = gy.Data[gyIndex];
                                    this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex++]);
                                    this.gb.Data[och] += gyData;
                                }
                            }
                        }
                    }
                }
            }

            return BatchArray.Convert(gx, x.Shape, x.BatchCount);
        }
    }
}
