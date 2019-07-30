using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "ReLU", Namespace = "KelpNet")]
    public class ReLU : SelectableSingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Initialize();
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
