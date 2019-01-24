using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class LeakyReLU<T> : CompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        private readonly Real<T> _slope;

        public LeakyReLU(double slope = 0.2f, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, name, inputNames, outputNames)
        {
            this._slope = slope;
        }

        internal override Real<T> ForwardActivate(Real<T> x)
        {
            return x < 0 ? x * this._slope : x;
        }

        internal override Real<T> BackwardActivate(Real<T> gy, Real<T> y)
        {
            return y <= 0 ? y * this._slope : gy;
        }
    }
}
