using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, bool gpuEnable = false) : base(name, gpuEnable, FUNCTION_NAME)
        {
        }

        public override Real ForwardActivate(Real x)
        {
            return x < 0 ? 0 : x;
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? 0 : gy;
        }
    }
}
