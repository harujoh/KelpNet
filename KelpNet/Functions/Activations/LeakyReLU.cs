using System;
using System.Collections.Generic;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class LeakyReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        private readonly Real _slope;

        public LeakyReLU(Real slope = 0.2f, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames)
        {
            this._slope = slope;
        }

        internal override Real ForwardActivate(Real x)
        {
            return x < 0 ? (Real)(x * this._slope) : x;
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? (Real)(y * this._slope) : gy;
        }
    }
}
