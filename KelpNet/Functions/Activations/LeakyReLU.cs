using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : Activation
    {
        const string FUNCTION_NAME = "LeakyReLU";

        private readonly Real _slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, bool isGpu = false) : base(name, isGpu)
        {
            this._slope = slope;

            this.ActivateFunctionString = Weaver.GetKernelSource(FUNCTION_NAME).Replace("/*slope*/", this._slope.ToString());

            if (IsGpu)
            {
                var program = Weaver.CreateProgram(ActivateFunctionString + ActivateKernelString);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
            }
        }

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x *= this._slope;
            }
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = y * this._slope;
            }
        }
    }
}
