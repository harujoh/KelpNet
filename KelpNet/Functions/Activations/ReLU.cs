using System;

namespace KelpNet
{
    [Serializable]
    public class ReLU<T> : CompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, null, name, inputNames, outputNames)
        {
        }

        internal override Real<T> ForwardActivate(Real<T> x)
        {
            return x < 0 ? 0 : x;
        }

        internal override Real<T> BackwardActivate(Real<T> gy, Real<T> y)
        {
            return y <= 0 ? 0 : gy;
        }
    }
}
