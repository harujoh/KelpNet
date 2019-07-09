using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class TanhActivation : CompressibleActivation
    {
        const string FUNCTION_NAME = "TanhActivation";

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override Real ForwardActivate(Real x)
        {
            return Math.Tanh(x);
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return gy * (1 - y * y);
        }
    }
}
