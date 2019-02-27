using System;
using KelpNet.Properties;

namespace KelpNet
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public override string ActivateFunctionString
        {
            get { return Weaver.GetKernelSource(Resources.Sigmoid); }
        }

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
        {
        }

        internal override Real ForwardActivate(Real x)
        {
            return Math.Tanh(x * 0.5) * 0.5 + 0.5;
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0 - y);
        }
    }
}
