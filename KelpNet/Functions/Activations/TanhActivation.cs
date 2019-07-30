using System;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "TanhActivation")]
    public class TanhActivation : SelectableSingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "TanhActivation";

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Initialize();
        }

        public Real ForwardActivate(Real x)
        {
            return Math.Tanh(x);
        }

        public Real BackwardActivate(Real gy, Real y)
        {
            return gy * (1 - y * y);
        }
    }
}
