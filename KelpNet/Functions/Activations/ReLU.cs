using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "ReLU", Namespace = "KelpNet")]
    public class ReLU : SingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
            return x < 0 ? 0 : x;
        }

        public Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? 0 : gy;
        }
    }
}
