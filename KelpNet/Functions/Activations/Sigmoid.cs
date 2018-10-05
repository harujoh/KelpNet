using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, null, name, inputNames, outputNames)
        {
        }

        internal override Real ForwardActivate(Real x)
        {
            return (Real)(1.0 / (1.0 + Math.Exp(-x)));
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0f - y);
        }
    }
}
