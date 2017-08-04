using System;
using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : Activation
    {
        public string ForwardKernelSource { get; }
        public string BackwardKernelSource { get; }

        private readonly Real _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU", bool isGpu = true) : base(name, isGpu)
        {
            this._slope = slope;

            if (IsGpu)
            {
                this.ForwardKernelSource = String.Format(this.ForwardActivateFunctionString, this._slope) + ForwardActivateKernelString;
                this.BackwardKernelSource = String.Format(this.BackwardActivateFunctionString, this._slope) + BackwardActivateKernelString;

                this.ForwardKernel = Weaver.CreateKernel(this.ForwardKernelSource, this.ForwardKernelName);
                this.BackwardKernel = Weaver.CreateKernel(this.BackwardKernelSource, this.BackwardKernelName);
            }
        }

        public override string ForwardActivateFunctionString { get; } =
@"
void ForwardActivate(__global Real* gpuY)
{{
    if(*gpuY < 0.0)
    {{
        *gpuY *= {0};
    }}
}}
";

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x *= this._slope;
            }
        }

        public override string BackwardActivateFunctionString { get; } =
@"
void BackwardActivate(Real gpuY, Real* gpugX)
{{
    if(gpuY <= 0.0)
    {{
        *gpugX = gpuY * {0};
    }}
}}";

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = y * this._slope;
            }
        }
    }
}
