using System;
using KelpNet.Properties;

namespace KelpNet
{
    [Serializable]
    public class TanhActivation : CompressibleActivation
    {
        const string FUNCTION_NAME = "TanhActivation";

        public override string ActivateFunctionString
        {
            get { return Weaver.GetKernelSource(Resources.TanhActivation); }
        }

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
        {
        }

        internal override Real ForwardActivate(Real x)
        {
            return Math.Tanh(x);
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return gy * (1 - y * y);
        }
    }
}
