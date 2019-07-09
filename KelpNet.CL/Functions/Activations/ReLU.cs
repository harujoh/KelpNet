using System;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class ReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";
        public override string ActivateFunctionString
        {
            get { return Weaver.GetKernelSource(Resources.ReLU); }
        }

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, null, name, inputNames, outputNames, gpuEnable)
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
