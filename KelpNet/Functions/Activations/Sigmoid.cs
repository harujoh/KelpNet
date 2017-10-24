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

        internal override Real ForwardActivate(Real x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1 - y);
        }
    }
}
