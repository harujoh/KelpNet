using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : NeedPreviousOutputFunction
    {
        public ReLU(string name = "ReLU") : base(name)
        {
            //カーネルを作成
            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, ForwardKernelName);
                //BackwardKernel = Weaver.CreateKernel("", "");
            }
        }

        const string ForwardKernelName = "ReLUForward";
        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void ReLUForward(
	__global double *gpuX,
	__global double *gpuY)
{
	int i = get_global_id(0);

    if(gpuX[i] > 0)
    {
        gpuY[i] = gpuX[i];
    }
    else
    {
        gpuY[i] = 0;
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] y = new double[x.Data.Length];

            if (!IsGpu)
            {
                for (int i = 0; i < x.Data.Length; i++)
                {
                    y[i] = Math.Max(0, x.Data[i]);
                }
            }
            else
            {
                ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data);
                ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y);

                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuY);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { x.Data.Length },
                    null,
                    null
                );

                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);

                gpuX.Dispose();
                gpuY.Dispose();
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            double[] gx = new double[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                gx[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
