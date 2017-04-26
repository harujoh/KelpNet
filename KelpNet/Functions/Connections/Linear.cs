using System;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        private readonly bool noBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Real[,] initialW = null, Real[] initialb = null, string name = "Linear", bool isGpu = true) : base(name, isGpu, inputCount, outputCount)
        {
            this.noBias = noBias;
            this.W = new NdArray(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                this.W.Data = initialW.Cast<Real>().ToArray();
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = new NdArray(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    this.b.Data = initialb.Cast<Real>().ToArray();
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        public override void InitKernel()
        {
            ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, "LinearForward");
            BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LinearBackward");
        }

        public override string ForwardKernelSource { get; } =
@"
__kernel void LinearForward(
	__global const Real *gpuX,
	__global const Real *gpuW, 
	__global Real *gpuY,
	const int OutputCount,
	const int InputCount)
{
	int batchCount = get_global_id(0);
	int i = get_global_id(1);

    gpuX += batchCount * InputCount;
    gpuW += i * InputCount;

    Real gpuYSum = 0;

    for (int j = 0; j < InputCount; j++)
    {
        gpuYSum = mad(gpuX[j], gpuW[j], gpuYSum);
    }
    
    gpuY[i + batchCount * OutputCount] += gpuYSum;
}";

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            Real[] y = new Real[OutputCount * x.BatchCount];

            if (!IsGpu)
            {
                for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        y[i + batchCount * this.OutputCount] += this.b.Data[i];

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            y[i + batchCount * this.OutputCount] += x.Data[j + batchCount * this.InputCount] * this.W.Data[i * this.InputCount + j];
                        }
                    }
                }
            }
            else
            {
                if (!this.noBias)
                {
                    for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
                    {
                        Array.Copy(this.b.Data, 0, y, batchCount * this.OutputCount, this.b.Data.Length);
                    }
                }

                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
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
                            new long[] { x.BatchCount, OutputCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            return BatchArray.Convert(y, new[] { OutputCount }, x.BatchCount);
        }

        public override string BackwardKernelSource { get; } =
@"
__kernel void LinearBackward(
	__global const Real *gpugY,
	__global const Real *gpuX,
	__global const Real *gpuW, 
	__global       Real *gpugW, 
	__global       Real *gpugb, 
	__global       Real *gpugX, 
	         const int BatchCount,
	         const int OutputCount,
	         const int InputCount)
{
    int j = get_global_id(0);

    gpugW += j;
    gpuW += j;

    gpugX += j;
    gpuX += j;

    for(int b = 0; b < BatchCount; b++)
    {
        for(int i = 0; i < OutputCount; i++)
        {
            Real gy = gpugY[i + b * OutputCount];

            if(j==0)
            {
                gpugb[i] += gy;
            }

            gpugW[i * InputCount] += gpuX[b * InputCount] * gy;
            gpugX[b * InputCount] += gpuW[i * InputCount] * gy;
        }
    }
}
";
        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            Real[] gxData = new Real[prevInput.Data.Length];

            if (!IsGpu)
            {
                for (int batchCount = 0; batchCount < gy.BatchCount; batchCount++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        Real gyData = gy.Data[i + batchCount * this.OutputCount];
                        this.gb.Data[i] += gyData;

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            this.gW.Data[i * this.InputCount + j] += prevInput.Data[j + batchCount * this.InputCount] * gyData;
                            gxData[j + batchCount * this.InputCount] += this.W.Data[i * this.InputCount + j] * gyData;
                        }
                    }
                }
            }
            else
            {
                using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, gy.Data))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevInput.Data))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                using (ComputeBuffer<Real> gpugb = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gb.Data))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gxData))
                {
                    BackwardKernel.SetMemoryArgument(0, gpugY);
                    BackwardKernel.SetMemoryArgument(1, gpuX);
                    BackwardKernel.SetMemoryArgument(2, gpuW);
                    BackwardKernel.SetMemoryArgument(3, gpugW);
                    BackwardKernel.SetMemoryArgument(4, gpugb);
                    BackwardKernel.SetMemoryArgument(5, gpugX);
                    BackwardKernel.SetValueArgument(6, gy.BatchCount);
                    BackwardKernel.SetValueArgument(7, this.OutputCount);
                    BackwardKernel.SetValueArgument(8, this.InputCount);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
                            null,
                            new long[] { this.InputCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugb, ref this.gb.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gxData, true, null);
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
