using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousInputFunction
    {
        private readonly Activation _activation;
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardgWKernel;

        [NonSerialized]
        public ComputeKernel BackwardgXKernel;

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

            this._activation = activation;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, int stride = 1, Size pad = new Size(), bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Conv2D", bool isGpu = true, Activation activation = null) : base(name, isGpu, inputChannels, outputChannels)
        {
            if (pad == Size.Empty)
            {
                pad = new Size(0, 0);
            }

            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._stride = stride;
            this._padX = pad.Width;
            this._padY = pad.Height;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this._activation = activation;

            this.Initialize(initialW, initialb);
        }

        void Initialize(Real[,,,] initialW = null, Real[] initialb = null)
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
                string forwardSource = this._activation != null ?
                                       this._activation.ForwardActivateFunctionString + this.ForwardKernelSource + "ForwardActivate(gpuY + index);}" :
                                       this.ForwardKernelSource + "}";

                this.ForwardKernel = Weaver.CreateProgram(forwardSource).CreateKernel("Convolution2DForward");
                this.BackwardgWKernel = Weaver.CreateProgram(this.BackwardgWKernelSource).CreateKernel("Convolution2DgWBackward");
                this.BackwardgXKernel = Weaver.CreateProgram(this.BackwardgXKernelSource).CreateKernel("Convolution2DgXBackward");
            }
        }

        public string ForwardKernelSource { get; } =
@"
__kernel void Convolution2DForward(
	const __global __read_only	Real* gpuX,
	const __global __read_only	Real* gpuW,
	const __global __read_only	Real* gpub,
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

	gpuW += och * InputCount * kHeight* kWidth;
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
//} Don't close for activation.
";

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
                                if (this._activation != null) this._activation.ForwardActivate(ref result[resultIndex]);
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
            if (this._activation != null)
            {
                this._prevOutput.Add(output);
            }

            return output;
        }

        string BackwardgWKernelSource { get; } =
@"
__kernel void Convolution2DgWBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuX,
		  __global __read_write Real* gpugW,
	const int batchCount,
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
	int och = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int ky = get_global_id(1);
	int kx = get_global_id(2);

	int outChOffset = och * inputCount * kHeight * kWidth;
	int gychOffset = och * gyShape1 * gyShape2;

	int iyStartIndex = ky - padY < 0 ? 0 : ky - padY;
	int iyLimit = gyShape1 * stride + ky - padY < xShape1 ? gyShape1 * stride + ky - padY : xShape1;

	int ixStartIndex = kx - padX < 0 ? 0 : kx - padX;
	int ixLimit = gyShape2 * stride + kx - padX < xShape2 ? gyShape2 * stride + kx - padX : xShape2;

	Real localgW = gpugW[outChOffset + ich * kHeight * kWidth + ky * kWidth + kx];

	for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
	{
		int gpuXIndex = batchCounter * xLength + ich * xShape1 * xShape2;
		int gyIndexOffset = batchCounter * gyShape0 * gyShape1 * gyShape2;

		for (int iy = iyStartIndex; iy < iyLimit; iy += stride)
		{
			int oy = iy - ky + padY;

			for (int ix = ixStartIndex; ix < ixLimit; ix += stride)
			{
				int ox = ix - kx + padX;

				int gyIndex = gyIndexOffset + gychOffset + oy * gyShape2 + ox;
				int inputIndex = gpuXIndex + iy * xShape2 + ix;

				localgW += gpuX[inputIndex] * activatedgy[gyIndex];
			}
		}
	}

	gpugW[outChOffset + ich * kHeight * kWidth + ky * kWidth + kx] = localgW;
}";


        string BackwardgXKernelSource { get; } =
@"
__kernel void Convolution2DgXBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuW,
		  __global __write_only Real* gpugX,
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
	int batchCounter = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int iy = get_global_id(1) + padY;
	int ix = get_global_id(2) + padX;

	int kyStart = 0 <= iy - gyShape1 * stride ? iy - gyShape1 * stride + 1 : 0;
	int kyLimit = kHeight < iy + 1 ? kHeight : iy + 1;

	int kxStart = 0 <= ix - gyShape2 * stride ? ix - gyShape2 * stride + 1 : 0;
	int kxLimit = kWidth < ix + 1 ? kWidth : ix + 1;

	Real localgX = 0;

	for (int och = 0; och < outputCount; och++)
	{
		int gyIndexOffset = batchCounter * gyShape0 * gyShape1 * gyShape2 + och * gyShape1 * gyShape2;
		int wIndexOffset = ich * kHeight * kWidth +  och * inputCount * kHeight * kWidth;

		for (int ky = kyStart; ky < kyLimit; ky++)
		{
			int kydiv = (iy - ky) / stride;

			for (int kx = kxStart; kx < kxLimit; kx++)
			{
				int kxdiv = (ix - kx) / stride;

				int gyIndex = gyIndexOffset + kydiv * gyShape2 + kxdiv;
				int wIndex = wIndexOffset + ky * kWidth + kx;

				localgX += gpuW[wIndex] * activatedgy[gyIndex];
			}
		}
	}

	gpugX[batchCounter * xLength + ich * xShape1 * xShape2 + (iy - padY) * xShape2 + (ix - padX)] = localgX;
}";

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray x)
        {
            Real[] prevOutputData = new Real[gy.Data.Length];
            if (this._activation != null)
            {
                prevOutputData = this._prevOutput[this._prevOutput.Count - 1].Data;
                this._prevOutput.RemoveAt(this._prevOutput.Count - 1);
            }

            Real[] gx = new Real[x.Data.Length];

            Real[] activatedgy = new Real[gy.BatchCount * gy.Length];
            for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
            {
                for (int och = 0; och < gy.Shape[0]; och++)
                {
                    for (int oy = 0; oy < gy.Shape[1]; oy++)
                    {
                        for (int ox = 0; ox < gy.Shape[2]; ox++)
                        {
                            int gyIndex = batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox;
                            Real gyData = gy.Data[gyIndex];
                            if (this._activation != null)
                            {
                                this._activation.BackwardActivate(ref gyData, prevOutputData[gyIndex]);
                            }
                            activatedgy[batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox] = gyData;

                            this.gb.Data[och] += gyData;
                        }
                    }
                }
            }

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

                                int gyIndex = batchCounter * gy.Length + och * gy.Shape[1] * gy.Shape[2] + oy * gy.Shape[2] + ox;

                                Real gyData = activatedgy[gyIndex]; //gyIndex = ch * ox * oy

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
                            }
                        }
                    }
                }
            }
            else
            {
                //gyは共通で使用
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
                {
                    using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                    using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                    {
                        this.BackwardgWKernel.SetMemoryArgument(0, gpugY);
                        this.BackwardgWKernel.SetMemoryArgument(1, gpuX);
                        this.BackwardgWKernel.SetMemoryArgument(2, gpugW);
                        this.BackwardgWKernel.SetValueArgument(3, gy.BatchCount);
                        this.BackwardgWKernel.SetValueArgument(4, this.InputCount);
                        this.BackwardgWKernel.SetValueArgument(5, gy.Shape[0]);
                        this.BackwardgWKernel.SetValueArgument(6, gy.Shape[1]);
                        this.BackwardgWKernel.SetValueArgument(7, gy.Shape[2]);
                        this.BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
                        this.BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
                        this.BackwardgWKernel.SetValueArgument(10, x.Length);
                        this.BackwardgWKernel.SetValueArgument(11, this._stride);
                        this.BackwardgWKernel.SetValueArgument(12, this._padX);
                        this.BackwardgWKernel.SetValueArgument(13, this._padY);
                        this.BackwardgWKernel.SetValueArgument(14, this._kHeight);
                        this.BackwardgWKernel.SetValueArgument(15, this._kWidth);

                        Weaver.CommandQueue.Execute
                        (
                            this.BackwardgWKernel,
                            null,
                            new long[] { OutputCount * InputCount, this._kHeight, this._kWidth },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    }

                    using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                    using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                    {
                        this.BackwardgXKernel.SetMemoryArgument(0, gpugY);
                        this.BackwardgXKernel.SetMemoryArgument(1, gpuW);
                        this.BackwardgXKernel.SetMemoryArgument(2, gpugX);
                        this.BackwardgXKernel.SetValueArgument(3, this.OutputCount);
                        this.BackwardgXKernel.SetValueArgument(4, this.InputCount);
                        this.BackwardgXKernel.SetValueArgument(5, gy.Shape[0]);
                        this.BackwardgXKernel.SetValueArgument(6, gy.Shape[1]);
                        this.BackwardgXKernel.SetValueArgument(7, gy.Shape[2]);
                        this.BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
                        this.BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
                        this.BackwardgXKernel.SetValueArgument(10, x.Length);
                        this.BackwardgXKernel.SetValueArgument(11, this._stride);
                        this.BackwardgXKernel.SetValueArgument(12, this._padX);
                        this.BackwardgXKernel.SetValueArgument(13, this._padY);
                        this.BackwardgXKernel.SetValueArgument(14, this._kHeight);
                        this.BackwardgXKernel.SetValueArgument(15, this._kWidth);

                        Weaver.CommandQueue.Execute
                        (
                            this.BackwardgXKernel,
                            null,
                            new long[] { gy.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
                            null,
                            null
                        );

                        Weaver.CommandQueue.Finish();
                        Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                    }
                }
            }

            return BatchArray.Convert(gx, x.Shape, x.BatchCount);
        }
    }
}
