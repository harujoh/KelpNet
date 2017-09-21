using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, bool gpuEnable = false) : base(name, gpuEnable, FUNCTION_NAME)
        {
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
