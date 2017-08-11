using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : Activation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, bool isGpu = false) : base(name, isGpu)
        {
            this.ActivateFunctionString = Weaver.GetKernelSource(FUNCTION_NAME);

            if (IsGpu)
            {
                var KernelSource = this.ActivateFunctionString + ActivateKernelString;

                var program = Weaver.CreateProgram(KernelSource);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
            }
        }

        public override void ForwardActivate(ref Real x)
        {
            x = 1 / (1 + Math.Exp(-x));
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            gy *= y * (1 - y);
        }
    }
}
