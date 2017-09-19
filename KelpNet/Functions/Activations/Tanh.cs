using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : CompressibleActivation
    {
        const string FUNCTION_NAME = "Tanh";

        public Tanh(string name = FUNCTION_NAME, bool isGpu = false) : base(name)
        {
            this.ActivateFunctionString = Weaver.GetKernelSource(FUNCTION_NAME);

            if (isGpu)
            {
                InitGpu();
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
