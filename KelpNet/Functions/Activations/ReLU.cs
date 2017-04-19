using System;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : NeedPreviousOutputFunction
    {
        public ReLU(string name = "ReLU",bool isGpu = true) : base(name, isGpu)
        {
        }

        public override void InitKernel()
        {
            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "ReLUForward");
            BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "ReLUBackward");
        }

        const string ForwardKernelSource =
@"
__kernel void ReLUForward(
	__global Real *gpuY)
{
	int i = get_global_id(0);

    if(gpuY[i] < 0.0)
    {
        gpuY[i] = 0.0;
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            Real[] y = x.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < x.Data.Length; i++)
                {
                    if (y[i] < 0)
                    {
                        y[i] = 0;
                    }
                }
            }
            else
            {
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
                {
                    ForwardKernel.SetMemoryArgument(0, gpuY);

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
__kernel void ReLUBackward(
	__global Real *gpuY,
	__global Real *gpugX)
{
	int i = get_global_id(0);

    if(gpuY[i] <= 0.0)
    {
        gpugX[i] = 0.0;
    }
}";
        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = gy.Data.ToArray();

            if (!IsGpu)
            {
                for (int i = 0; i < gy.Data.Length; i++)
                {
                    if (prevOutput.Data[i] <= 0)
                    {
                        gx[i] = 0;
                    }
                }
            }
            else
            {
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, prevOutput.Data))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, gx))
                {
                    BackwardKernel.SetMemoryArgument(0, gpuY);
                    BackwardKernel.SetMemoryArgument(1, gpugX);

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
