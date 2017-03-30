using System;
using System.Collections.Generic;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double dropoutRatio;
        private readonly List<double[]> maskStack = new List<double[]>();

        public Dropout(double dropoutRatio = 0.5, string name = "Dropout", bool isGpu = true) : base(name, isGpu)
        {
            this.dropoutRatio = dropoutRatio;
        }

        public override void InitKernel()
        {
            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, "DropoutForward");
            BackwardKernel = Weaver.CreateKernel(BackwardKernelSource, "DropoutBackward");
        }

        const string ForwardKernelSource =
@"
#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel void DropoutForward(
	__global const double *gpuX,
	__global const double *mask,
	__global double *gpuY,
    int maskLength)
{
	int i = get_global_id(0);

    gpuY[i] = gpuX[i] * mask[i % maskLength];
}";

        protected override BatchArray ForwardSingle(BatchArray x)
        {
            double[] result = new double[x.Data.Length];
            double[] mask = new double[x.Length];
            double scale = 1.0 / (1.0 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            if (!IsGpu)
            {
                for (int i = 0; i < x.Data.Length; i++)
                {
                    result[i] = x.Data[i] * mask[i % mask.Length];
                }
            }
            else
            {
                using (ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<double> gpuMask = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
                using (ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.WriteOnly, result.Length))
                {
                    ForwardKernel.SetMemoryArgument(0, gpuX);
                    ForwardKernel.SetMemoryArgument(1, gpuMask);
                    ForwardKernel.SetMemoryArgument(2, gpuY);
                    ForwardKernel.SetValueArgument(3, mask.Length);

                    Weaver.CommandQueue.Execute
                        (
                            ForwardKernel,
                            null,
                            new long[] { x.Data.Length },
                            null,
                            null
                        );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);
                }
            }

            this.maskStack.Add(mask);

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        const string BackwardKernelSource =
@"
#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel void DropoutBackward(
	__global const double *mask,
	__global double *gpugX,
    int gyLength)
{
	int b = get_global_id(0);
	int j = get_global_id(1);

    gpugX[j + b * gyLength] *= mask[j];
}";

        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            double[] result = gy.Data.ToArray();
            double[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            if (!IsGpu)
            {
                for (int b = 0; b < gy.BatchCount; b++)
                {
                    for (int j = 0; j < mask.Length; j++)
                    {
                        result[j + b * gy.Length] *= mask[j];
                    }
                }
            }
            else
            {
                using (ComputeBuffer<double> gpuMask = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
                using (ComputeBuffer<double> gpugX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, result))
                {
                    BackwardKernel.SetMemoryArgument(0, gpuMask);
                    BackwardKernel.SetMemoryArgument(1, gpugX);
                    BackwardKernel.SetValueArgument(2, gy.Length);

                    Weaver.CommandQueue.Execute
                    (
                        BackwardKernel,
                        null,
                        new long[] { gy.BatchCount, mask.Length },
                        null,
                        null
                    );

                    Weaver.CommandQueue.Finish();
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref result, true, null);
                }
            }
            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }

        //Predict時に何もしない
        public override BatchArray Predict(BatchArray input)
        {
            return input;
        }
    }
}
