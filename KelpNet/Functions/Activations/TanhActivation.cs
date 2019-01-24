using System;

namespace KelpNet
{
    [Serializable]
    public class TanhActivation<T> : CompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "TanhActivation";

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, null, name, inputNames, outputNames)
        {
        }

        internal override Real<T> ForwardActivate(Real<T> x)
        {
            return Math.Tanh(x);
        }

        internal override Real<T> BackwardActivate(Real<T> gy, Real<T> y)
        {
            return gy * (1.0f - y * y);
        }
    }
}
