using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override Real ForwardActivate(Real x)
        {
            return Math.Tanh(x * 0.5) * 0.5 + 0.5;
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0 - y);
        }
    }
}
