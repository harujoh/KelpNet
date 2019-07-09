using System;

namespace KelpNet.CPU
{
    [Serializable]
    public class LeakyReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";

        private readonly Real _slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
