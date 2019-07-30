using System;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "Sigmoid")]
    public class Sigmoid : SelectableSingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Initialize();
        }

        public Real ForwardActivate(Real x)
        {
            return Math.Tanh(x * 0.5) * 0.5 + 0.5;
        }

        public Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0 - y);
        }
    }
}
