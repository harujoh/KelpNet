using System;
using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : Activation
    {
        public override string ForwardKernelSource { get; }
        public override string BackwardKernelSource { get; }

        public override string ForwardActivateGPU { get; } =
@"
void ForwardActivate(__global Real* gpuY)
{
    if(*gpuY < 0.0)
    {
        *gpuY = 0.0;
    }
}
";

        public override string BackwardActivateGPU { get; } =
@"
void BackwardActivate(Real gpuY, __global Real* gpugX)
{
    if(gpuY <= 0.0)
    {
        *gpugX = 0.0;
    }
}
";

        public ReLU(string name = "ReLU", bool isGpu = true) : base(name, isGpu)
        {
            if (IsGpu)
            {
                this.ForwardKernelSource = ForwardActivateGPU + String.Format(ForwardKernelString, this.ForwardKernelName);
                this.BackwardKernelSource = BackwardActivateGPU + String.Format(BackwardKernelString, this.BackwardKernelName);

                this.ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, this.ForwardKernelName);
                this.BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, this.BackwardKernelName);
            }
        }

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x = 0;
            }
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = 0;
            }
        }
    }
}
