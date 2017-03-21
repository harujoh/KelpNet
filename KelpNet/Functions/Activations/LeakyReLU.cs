using System;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousOutputFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU") : base(name)
        {
            this._slope = slope;

            //カーネルを作成
            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "LeakyReLUForward");
                BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "LeakyReLUBackward");
            }
        }

        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LeakyReLUForward(
	         const double slope,
	__global double *gpuY)
{
	int i = get_global_id(0);

    if(gpuY[i] < 0.0)
    {
        gpuY[i] *= slope;
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] y = x.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < x.Data.Length; i++)
                {
                    if (y[i] < 0) y[i] *= this._slope;
                }
            }
            else
            {
                using (ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, y))
                {
                    ForwardKernel.SetValueArgument(0, this._slope);
                    ForwardKernel.SetMemoryArgument(1, gpuY);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
                            null,
                            new long[] { x.Data.Length },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
                }
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        const string BackwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void LeakyReLUBackward(
	__global double *gpuY,
  	   const double slope,
	__global double *gpugX)
{
	int i = get_global_id(0);

    if(gpuY[i] <= 0.0)
    {
        gpugX[i] = gpuY[i] * slope;
    }
}";

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            double[] gx = gy.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < gy.Data.Length; i++)
                {
                    if (prevOutput.Data[i] <= 0)
                    {
                        gx[i] = prevOutput.Data[i] * this._slope;
                    }
                }
            }
            else
            {
                using (ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutput.Data))
                using (ComputeBuffer<double> gpugX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, gx))
                {
                    BackwardKernel.SetMemoryArgument(0, gpuY);
                    ForwardKernel.SetValueArgument(1, this._slope);
                    BackwardKernel.SetMemoryArgument(2, gpugX);

                    Weaver.CommandQueue.Execute
                        (
                            BackwardKernel,
                            null,
                            new long[] { gy.Data.Length },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
