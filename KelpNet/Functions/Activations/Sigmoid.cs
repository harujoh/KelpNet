using System;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "Sigmoid", Namespace = "KelpNet")]
    public class Sigmoid : SingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            return this.NeedPreviousForwardCpu(x);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            this.NeedPreviousBackwardCpu(y, x);
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
