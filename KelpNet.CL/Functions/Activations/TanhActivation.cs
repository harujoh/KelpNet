using System;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class TanhActivation : CompressibleActivation
    {
        const string FUNCTION_NAME = "TanhActivation";

        public override string ActivateFunctionString
        {
            get { return OpenCL.GetKernelSource(Resources.TanhActivation); }
        }

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
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
