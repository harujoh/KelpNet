using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        private readonly Real _slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, bool gpuEnable = false) : base(name, gpuEnable, FUNCTION_NAME, new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()))
        {
            this._slope = slope;
        }

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x *= this._slope;
            }
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = y * this._slope;
            }
        }
    }
}
