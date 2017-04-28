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
        private readonly Real dropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        public Dropout(Real? dropoutRatio = null, string name = "Dropout", bool isGpu = true) : base(name, isGpu)
        {
            this.dropoutRatio = dropoutRatio ?? (Real)0.5;

            if (IsGpu)
            {
                ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, "DropoutForward");
                BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, "DropoutBackward");
            }
        }

        public override string ForwardKernelSource { get; } =
@"
__kernel void DropoutForward(
	__global const Real *gpuX,
	__global const Real *mask,
	__global Real *gpuY,
    int maskLength)
{
	int i = get_global_id(0);

    gpuY[i] = gpuX[i] * mask[i % maskLength];
}";

        protected override BatchArray ForwardSingle(BatchArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = new Real[x.Length];
            Real scale = 1 / (1 - this.dropoutRatio);

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
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
                using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, result.Length))
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

        public override string BackwardKernelSource { get; } =
@"
__kernel void DropoutBackward(
	__global const Real *mask,
	__global Real *gpugX,
    int gyLength)
{
	int b = get_global_id(0);
	int j = get_global_id(1);

    gpugX[j + b * gyLength] *= mask[j];
}";

        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            Real[] result = gy.Data.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
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
                using (ComputeBuffer<Real> gpuMask = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, mask))
                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, result))
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
