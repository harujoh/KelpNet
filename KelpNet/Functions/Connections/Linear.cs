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

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name, inputCount, outputCount)
        {
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
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, ForwardKernelName);
                //BackwardKernel = Weaver.CreateKernel("", "");
            }
        }

        const string ForwardKernelName = "LinearForward";
        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LinearForward(
	__global double *gpuX,
	__global double *gpuW, 
	__global double *gpub, 
	__global double *gpuY,
	int OutputCount,
	int InputCount)
{
	int batchCount = get_global_id(0);
	int i = get_global_id(1);

    gpuY[i + batchCount * OutputCount] += gpub[i];

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
                            y[i + batchCount * this.OutputCount] += x.Data[j + batchCount * this.InputCount] *
                                                                       this.W.Data[i * this.InputCount + j];
                        }
                    }
                }
            }
            else
            {
                ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data);
                ComputeBuffer<double> gpuW = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.W.Data);
                ComputeBuffer<double> gpub = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.b.Data);
                ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y);

                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuW);
                ForwardKernel.SetMemoryArgument(2, gpub);
                ForwardKernel.SetMemoryArgument(3, gpuY);
                ForwardKernel.SetValueArgument(4, this.OutputCount);
                ForwardKernel.SetValueArgument(5, this.InputCount);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { x.BatchCount, OutputCount },
                    null,
                    null
                );

                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);

                gpuX.Dispose();
                gpuW.Dispose();
                gpub.Dispose();
                gpuY.Dispose();
            }

            return BatchArray.Convert(y, new[] { OutputCount }, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            double[] gxData = new double[prevInput.Data.Length];

            for (int b = 0; b < gy.BatchCount; b++)
            {
                int indexOffset = 0;

                for (int i = 0; i < gy.Length; i++)
                {
                    double gyData = gy.Data[i + b * gy.Length];
                    this.gb.Data[i] += gyData;

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.gW.Data[indexOffset] += prevInput.Data[j + b * prevInput.Length] * gyData;
                        gxData[j + b * prevInput.Length] += this.W.Data[indexOffset++] * gyData;
                    }
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
