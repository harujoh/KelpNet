using System;
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

        private bool noBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name, inputCount, outputCount)
        {
            this.noBias = noBias;
            this.W = NdArray.Zeros(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }

            //カーネルを作成
            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "LinearForward");
                ForwardKernel.SetValueArgument(3, this.OutputCount);
                ForwardKernel.SetValueArgument(4, this.InputCount);

                BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LinearBackward");
                BackwardKernel.SetValueArgument(6, this.OutputCount);
                BackwardKernel.SetValueArgument(7, this.InputCount);
            }
        }

        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LinearForward(
	__global double *gpuX,
	__global double *gpuW, 
	__global double *gpuY,
	int OutputCount,
	int InputCount)
{
	int batchCount = get_global_id(0);
	int i = get_global_id(1);

    for (int j = 0; j < InputCount; j++)
    {
        gpuY[i + batchCount * OutputCount] += gpuX[j + batchCount * InputCount] * gpuW[i * InputCount + j];
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] y = new double[OutputCount * x.BatchCount];

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

                using (ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<double> gpuW = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
                {
                    ForwardKernel.SetMemoryArgument(0, gpuX);
                    ForwardKernel.SetMemoryArgument(1, gpuW);
                    ForwardKernel.SetMemoryArgument(2, gpuY);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
                            null,
                            new long[] { x.BatchCount, OutputCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            return BatchArray.Convert(y, new[] { OutputCount }, x.BatchCount);
        }

        const string BackwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LinearBackward(
	__global double *gpugY,
	__global double *gpuX,
	__global double *gpuW, 
	__global double *gpugW, 
	__global double *gpugb, 
	__global double *gpugX, 
	int OutputCount,
	int InputCount)
{
	int b = get_global_id(0);
	int i = get_global_id(1);

    int indexOffset = i * InputCount;

    double gyData = gpugY[i + b * OutputCount];
    gpugb[i] += gyData;

    for (int j = 0; j < InputCount; j++)
    {
        gpugW[indexOffset] += gpuX[j + b * InputCount] * gyData;
        gpugX[j + b * InputCount] += gpuW[indexOffset + j] * gyData;
    }
}";

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            double[] gxData = new double[prevInput.Data.Length];

            if (!IsGpu)
            {
                for (int b = 0; b < gy.BatchCount; b++)
                {
                    int indexOffset = 0;

                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        double gyData = gy.Data[i + b * this.OutputCount];
                        this.gb.Data[i] += gyData;

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            this.gW.Data[indexOffset] += prevInput.Data[j + b * this.InputCount] * gyData;
                            gxData[j + b * this.InputCount] += this.W.Data[indexOffset++] * gyData;
                        }
                    }
                }
            }
            else
            {
                using (ComputeBuffer<double> gpugY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, gy.Data))
                using (ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevInput.Data))
                using (ComputeBuffer<double> gpuW = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data))
                using (ComputeBuffer<double> gpugW = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gW.Data))
                using (ComputeBuffer<double> gpugb = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.gb.Data))
                using (ComputeBuffer<double> gpugX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gxData))
                {
                    BackwardKernel.SetMemoryArgument(0, gpugY);
                    BackwardKernel.SetMemoryArgument(1, gpuX);
                    BackwardKernel.SetMemoryArgument(2, gpuW);
                    BackwardKernel.SetMemoryArgument(3, gpugW);
                    BackwardKernel.SetMemoryArgument(4, gpugb);
                    BackwardKernel.SetMemoryArgument(5, gpugX);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
                            null,
                            new long[] { gy.BatchCount, OutputCount },
                            null,
                            null
                        );

                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.gW.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugb, ref this.gb.Data, true, null);
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gxData, true, null);
                }

            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
