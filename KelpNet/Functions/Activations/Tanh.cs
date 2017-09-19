using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : Activation
    {
        const string FUNCTION_NAME = "Tanh";

        public Tanh(string name = FUNCTION_NAME, bool isGpu = false) : base(name, isGpu)
        {
            this.ActivateFunctionString = Weaver.GetKernelSource(FUNCTION_NAME);

            if (IsGpu)
            {
                string kernelSource = this.ActivateFunctionString + ActivateKernelString;

                ComputeProgram program = Weaver.CreateProgram(kernelSource);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
            }
        }

        public override void ForwardActivate(ref Real x)
        {
            x = Math.Tanh(x);
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            gy *= 1 - y * y;
        }
    }
}
