using System;
using System.Collections.Generic;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class LeakyReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        public Real Slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.LeakyReLU), new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames, gpuEnable)
        {
            this.Slope = slope;
        }

        public override Real ForwardActivate(Real x)
        {
            return x < 0 ? (Real)(x * this.Slope) : x;
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? (Real)(y * this.Slope) : gy;
        }
    }
}
