using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, bool isGpu = false) : base(name)
        {
            this.ActivateFunctionString = Weaver.GetKernelSource(FUNCTION_NAME);

            if (isGpu)
            {
                InitGpu();
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
