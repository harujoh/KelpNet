using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : Activation
    {
        public string ForwardKernelSource { get; }
        public string BackwardKernelSource { get; }

        public Tanh(string name = "Tanh", bool isGpu = true) : base(name, isGpu)
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
    *gpuY = tanh(*gpuY);
}
";

        public override void ForwardActivate(ref Real x)
        {
            x = Math.Tanh(x);
        }

        public override string BackwardActivateFunctionString { get; } =
@"
void BackwardActivate(Real gpuY, Real* gpugX)
{
    *gpugX *= 1 - gpuY * gpuY;
}
";

        public override void BackwardActivate(ref Real gy, Real y)
        {
            gy *= 1 - y * y;
        }
    }
}
