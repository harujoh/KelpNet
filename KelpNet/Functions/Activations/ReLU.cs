using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class ReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
