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

        private readonly bool noBias;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear", bool isGpu = true) : base(name, inputCount, outputCount)
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
            if (isGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "LinearForward");
                BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LinearBackward");
            }
        }

        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LinearForward(
	__global const double *gpuX,
	__global const double *gpuW, 
	__global double *gpuY,
	const int OutputCount,
	const int InputCount)
{
	int batchCount = get_global_id(0);
	int i = get_global_id(1);

    for (int j = 0; j < InputCount; j++)
    {
        gpuY[i + batchCount * OutputCount] += gpuX[j + batchCount * InputCount] * gpuW[i * InputCount + j];
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray x, bool isGpu)
        {
            double[] y = new double[OutputCount * x.BatchCount];

            if (!isGpu)
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

        const string BackwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

double atom_add_double(__global double* const address, const double value)
{
  long oldval, newval, readback;
  
  *(double*)&oldval = *address;
  *(double*)&newval = (*(double*)&oldval + value);
  while ((readback = atom_cmpxchg((__global long*)address, oldval, newval)) != oldval) {
    oldval = readback;
    *(double*)&newval = (*(double*)&oldval + value);
  }
  return *(double*)&oldval;
}

__kernel void LinearBackward(
	__global const double *gpugY,
	__global const double *gpuX,
	__global const double *gpuW, 
	__global       double *gpugW, 
	__global       double *gpugb, 
	__global       double *gpugX, 
	         const int OutputCount,
	         const int InputCount)
{
    int b = get_global_id(0);
    int i = get_global_id(1);

    atom_add_double(&gpugb[i], gpugY[i + b * OutputCount]);

    for (int j = 0; j < InputCount; j++)
    {
        atom_add_double(&gpugW[i * InputCount + j], gpuX[j + b * InputCount] * gpugY[i + b * OutputCount]);
        atom_add_double(&gpugX[j + b * InputCount], gpuW[i * InputCount + j] * gpugY[i + b * OutputCount]);
    }
}
";
        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput, bool isGpu)
        {
            double[] gxData = new double[prevInput.Data.Length];

            if (!isGpu)
            {
                for (int b = 0; b < gy.BatchCount; b++)
                {
                    for (int i = 0; i < this.OutputCount; i++)
                    {
                        double gyData = gy.Data[i + b * this.OutputCount];
                        this.gb.Data[i] += gyData;

                        for (int j = 0; j < this.InputCount; j++)
                        {
                            this.gW.Data[i * this.InputCount + j] += prevInput.Data[j + b * this.InputCount] * gyData;
                            gxData[j + b * this.InputCount] += this.W.Data[i * this.InputCount + j] * gyData;
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
                    BackwardKernel.SetValueArgument(6, this.OutputCount);
                    BackwardKernel.SetValueArgument(7, this.InputCount);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
                            null,
                            new long[] { gy.BatchCount, this.OutputCount },
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
