using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
        {
        }

        internal override Real ForwardActivate(Real x)
        {
            return x < 0 ? 0 : x;
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? 0 : gy;
        }
    }
}
