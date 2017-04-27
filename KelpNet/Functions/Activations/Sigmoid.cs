using System;
using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : Activation
    {
        public override string ForwardKernelSource { get; }
        public override string BackwardKernelSource { get; }

        public override string ForwardActivateGPU { get; } =
@"
void ForwardActivate(__global Real* gpuY)
{
    *gpuY = 1 / (1 + exp(-*gpuY));
}
";

        public override string BackwardActivateGPU { get; } =
@"
void BackwardActivate(Real gpuY, __global Real* gpugX)
{
    *gpugX *= gpuY * (1 - gpuY);
}
";

        public Sigmoid(string name = "Sigmoid", bool isGpu = true) : base(name, isGpu)
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
            x = 1 / (1 + (Real)Math.Exp(-x));
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            gy *= y * (1 - y);
        }
    }
}
