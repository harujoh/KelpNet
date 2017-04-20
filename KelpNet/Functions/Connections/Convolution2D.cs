using System;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Convolution2D : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _stride;
        private readonly int _padX;
        private readonly int _padY;

        public Convolution2D(int inputChannels, int outputChannels, int kSize, int stride = 1, int pad = 0, bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Conv2D", bool isGpu = true) : base(name, isGpu, inputChannels, outputChannels)
        {
            this._kWidth = kSize;
            this._kHeight = kSize;
            this._stride = stride;
            this._padX = pad;
            this._padY = pad;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this.Initialize(initialW, initialb, isGpu);
        }

        public Convolution2D(int inputChannels, int outputChannels, Size kSize, int stride = 1, Size pad = new Size(), bool noBias = false, Real[,,,] initialW = null, Real[] initialb = null, string name = "Conv2D", bool isGpu = true) : base(name, isGpu, inputChannels, outputChannels)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kWidth = kSize.Width;
            this._kHeight = kSize.Height;
            this._stride = stride;
            this._padX = pad.Width;
            this._padY = pad.Height;

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            this.Initialize(initialW, initialb, isGpu);
        }

        void Initialize(Real[,,,] initialW = null, Real[] initialb = null, bool isGpu = true)
        {
            this.W = NdArray.Zeros(OutputCount, InputCount, this._kHeight, this._kWidth);
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
            this.b = NdArray.Zeros(OutputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (this.Parameters.Length > 1)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        public override void InitKernel()
        {
            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "Convolution2DForward");
            BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "Convolution2DBackward");
        }

        const string ForwardKernelSource =
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
    int oy = get_global_id(1);
    int ox = get_global_id(2);

    Real localResult = 0;

    gpuW += och * InputCount * kHeight * kWidth;
    gpuX += batchCounter * inputLength;

    for (int ich = 0; ich < InputCount; ich++)
    {
        for (int ky = 0; ky < kHeight; ky++)
        {
            int iy = oy * stride + ky - padY;

            if (iy >= 0 && iy < inputShape1)
            {
                for (int kx = 0; kx < kWidth; kx++)
                {
                    int ix = ox * stride + kx - padX;

                    if (ix >= 0 && ix < inputShape2)
                    {
                        int inputIndex = ich * inputShape1 * inputShape2 + iy * inputShape2 + ix;
                        int wIndex = ich * kHeight * kWidth + ky * kWidth + kx;

                        localResult += gpuX[inputIndex] * gpuW[wIndex];
                    }
                }
            }
        }
    }

    gpuY[batchCounter * OutputCount * outputHeight * outputWidth + och * outputHeight * outputWidth + oy * outputWidth + ox] = localResult + gpub[och];
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

                        for (int oy = 0; oy < outputHeight; oy++)
                        {
                            for (int ox = 0; ox < outputWidth; ox++)
                            {
                                for (int ich = 0; ich < this.InputCount; ich++)
                                {
                                    //Wインデックス用
                                    int inChOffset = ich * this._kHeight * this._kWidth;

                                    //inputインデックス用
                                    int inputOffset = ich * input.Shape[1] * input.Shape[2];

                                    for (int ky = 0; ky < this._kHeight; ky++)
                                    {
                                        int iy = oy * this._stride + ky - this._padY;

                                        if (iy >= 0 && iy < input.Shape[1])
                                        {
                                            for (int kx = 0; kx < this._kWidth; kx++)
                                            {
                                                int ix = ox * this._stride + kx - this._padX;

                                                if (ix >= 0 && ix < input.Shape[2])
                                                {
                                                    int wIndex = outChOffset + inChOffset + ky * this._kWidth + kx;
                                                    int inputIndex = inputOffset + iy * input.Shape[2] + ix + batchCounter * input.Length;

                                                    result[resultIndex] += input.Data[inputIndex] * this.W.Data[wIndex];
                                                }
                                            }
                                        }
                                    }
                                }

                                result[resultIndex] += this.b.Data[och];
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

            return BatchArray.Convert(result, new[] { this.OutputCount, outputHeight, outputWidth }, input.BatchCount);
        }


        const string BackwardKernelSource =
@"
__kernel void Convolution2DBackward(
	const __global __read_only  Real* gpugY,
	const __global __read_only  Real* gpuX,
	const __global __read_only  Real* gpuW, 
  	      __global __read_write Real* gpugW, 
	      __global __read_write Real* gpugb, 
	      __global __read_write Real* gpugX, 
	      __global __read_write Real* tmpgW, 
	      __local  Real* lgW,
	const int OutputCount,
	const int InputCount,
	const int BatchCount,
    const int gyShape0,
    const int gyShape1,
    const int gyShape2,
    const int xShape0,
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

    int wLength = OutputCount * InputCount * kHeight * kWidth;

    gpuW += ich * kHeight * kWidth;
    //lgW += ich * kHeight * kWidth;
    gpuX += ich * xShape1 * xShape2; 
    gpugX += ich * xShape1 * xShape2; 

    //for (int batchCounter = 0; batchCounter < BatchCount; batchCounter++)
    {
        for (int och = 0; och < gyShape0; och++)
        {
            for (int oy = 0; oy < gyShape1; oy++)
            {
                for (int ox = 0; ox < gyShape2; ox++)
                {
                    Real gyData = gpugY[batchCounter * gyShape0 * gyShape1 * gyShape2 + och * gyShape1 * gyShape2 + oy * gyShape2 + ox];

                    for (int ky = 0; ky < kHeight; ky++)
                    {
                        int iy = oy * stride + ky - padY;

                        if (iy >= 0 && iy < xShape1)
                        {
                            for (int kx = 0; kx < kWidth; kx++)
                            {
                                int ix = ox * stride + kx - padX;

                                if (ix >= 0 && ix < xShape2)
                                {
                                    int wIndex = och * InputCount * kHeight * kWidth + ky * kWidth + kx;
                                    int inputIndex = batchCounter * xLength + iy * xShape2 + ix;

                                    lgW[wIndex + ich * kHeight * kWidth] += gpuX[inputIndex] * gyData;
                                    gpugX[inputIndex] += gpuW[wIndex] * gyData;
                                }
                            }
                        }
                    }

                    if(ich == 0)
                    {
                        gpugb[och] += gyData;
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0;i < wLength; i++)
    {
        tmpgW[batchCounter * wLength + i] = lgW[i];
    }

	barrier(CLK_GLOBAL_MEM_FENCE);

    if(batchCounter == 0 && ich==0)
    {
        for(int bc = 0; bc < BatchCount; bc++)
        {
            for(int i = 0; i < wLength; i++)
            {
                gpugW[i] = tmpgW[bc * wLength + i];
            }
        }
    }
}";

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray x)
        {
            Real[] gx = new Real[x.Data.Length];

            if (!IsGpu)
            {
                int gyIndex = 0;

                for (int batchCounter = 0; batchCounter < gy.BatchCount; batchCounter++)
                {
                    for (int och = 0; och < gy.Shape[0]; och++)
                    {
                        //gWインデックス用
                        int outChOffset = och * this.InputCount * this._kHeight * this._kWidth;

                        for (int oy = 0; oy < gy.Shape[1]; oy++)
                        {
                            for (int ox = 0; ox < gy.Shape[2]; ox++)
                            {
                                Real gyData = gy.Data[gyIndex++]; //gyIndex = ch * ox * oy

                                for (int ich = 0; ich < x.Shape[0]; ich++)
                                {
                                    //gWインデックス用
                                    int inChOffset = ich * this._kHeight * this._kWidth;

                                    //inputインデックス用
                                    int inputOffset = ich * x.Shape[1] * x.Shape[2] + batchCounter * x.Length;

                                    for (int ky = 0; ky < this._kHeight; ky++)
                                    {
                                        int iy = oy * this._stride + ky - this._padY;

                                        if (iy >= 0 && iy < x.Shape[1])
                                        {
                                            for (int kx = 0; kx < this._kWidth; kx++)
                                            {
                                                int ix = ox * this._stride + kx - this._padX;

                                                if (ix >= 0 && ix < x.Shape[2])
                                                {
                                                    //WとgWのshapeは等しい
                                                    int wIndex = outChOffset + inChOffset + ky * this._kWidth + kx;

                                                    //xとgxのshapeは等しい
                                                    int inputIndex = inputOffset + iy * x.Shape[2] + ix;

                                                    this.gW.Data[wIndex] += x.Data[inputIndex] * gyData;

                                                    gx[inputIndex] += this.W.Data[wIndex] * gyData;
                                                }
                                            }
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
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, gy.Data))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                using (ComputeBuffer<Real> gpugb = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gb.Data))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gx))
                using (ComputeBuffer<Real> tmpgW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.AllocateHostPointer, this.gW.Data.Length * gy.BatchCount))
                {
                    BackwardKernel.SetMemoryArgument(0, gpugY);
                    BackwardKernel.SetMemoryArgument(1, gpuX);
                    BackwardKernel.SetMemoryArgument(2, gpuW);
                    BackwardKernel.SetMemoryArgument(3, gpugW);
                    BackwardKernel.SetMemoryArgument(4, gpugb);
                    BackwardKernel.SetMemoryArgument(5, gpugX);
                    BackwardKernel.SetMemoryArgument(6, tmpgW);
                    BackwardKernel.SetLocalArgument(7, this.W.Data.Length * Marshal.SizeOf(typeof(Real)));
                    BackwardKernel.SetValueArgument(8, this.OutputCount);
                    BackwardKernel.SetValueArgument(9, this.InputCount);
                    BackwardKernel.SetValueArgument(10, gy.BatchCount);
                    BackwardKernel.SetValueArgument(11, gy.Shape[0]);
                    BackwardKernel.SetValueArgument(12, gy.Shape[1]);
                    BackwardKernel.SetValueArgument(13, gy.Shape[2]);
                    BackwardKernel.SetValueArgument(14, x.Shape[0]);
                    BackwardKernel.SetValueArgument(15, x.Shape[1]);
                    BackwardKernel.SetValueArgument(16, x.Shape[2]);
                    BackwardKernel.SetValueArgument(17, x.Length);
                    BackwardKernel.SetValueArgument(18, this._stride);
                    BackwardKernel.SetValueArgument(19, this._padX);
                    BackwardKernel.SetValueArgument(20, this._padY);
                    BackwardKernel.SetValueArgument(21, this._kHeight);
                    BackwardKernel.SetValueArgument(22, this._kWidth);

                    Weaver.CommandQueue.Execute
                    (
                        BackwardKernel,
                        null,
                        new long[] { gy.BatchCount, x.Shape[0] },
                        new long[] { gy.BatchCount, 1 },
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugb, ref this.gb.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            return BatchArray.Convert(gx, x.Shape, x.BatchCount);
        }
    }
}
