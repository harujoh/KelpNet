using System;
using System.Collections.Generic;
using KelpNet.CL.Properties;

namespace KelpNet.CL.Activations
{
    [Serializable]
    public class LeakyReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        public override string ActivateFunctionString
        {
            get { return Weaver.GetKernelSource(Resources.LeakyReLU); }
        }

        private readonly Real _slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames, gpuEnable)
        {
            this._slope = slope;
        }

        public override Real ForwardActivate(Real x)
        {
            return x < 0 ? (Real)(x * this._slope) : x;
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? (Real)(y * this._slope) : gy;
        }
    }
}
