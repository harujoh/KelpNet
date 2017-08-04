using System;
using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : Activation
    {
        public string ForwardKernelSource { get; }
        public string BackwardKernelSource { get; }

        public ReLU(string name = "ReLU", bool isGpu = true) : base(name, isGpu)
        {
            if (IsGpu)
            {
                this.ForwardKernelSource = this.ForwardActivateFunctionString + ForwardActivateKernelString;
                this.BackwardKernelSource = this.BackwardActivateFunctionString + BackwardActivateKernelString;

                this.ForwardKernel = Weaver.CreateProgram(this.ForwardKernelSource).CreateKernel(this.ForwardKernelName);
                this.BackwardKernel = Weaver.CreateProgram(this.BackwardKernelSource).CreateKernel(this.BackwardKernelName);
            }
        }

        public override string ForwardActivateFunctionString { get; } =
@"
void ForwardActivate(__global Real* gpuY)
{
    if(*gpuY < 0.0)
    {
        *gpuY = 0.0;
    }
}
";

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x = 0;
            }
        }

        public override string BackwardActivateFunctionString { get; } =
@"
void BackwardActivate(Real gpuY, Real* gpugX)
{
    if(gpuY <= 0.0)
    {
        *gpugX = 0.0;
    }
}
";

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = 0;
            }
        }
    }
}
