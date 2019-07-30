using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "LeakyReLU", Namespace = "KelpNet")]
    public class LeakyReLU : SelectableSingleInputFunction, ICompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";

        [DataMember]
        public Real Slope;

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Slope = slope;

            this.Initialize();
        }

        public Real ForwardActivate(Real x)
        {
            return x < 0 ? (Real)(x * this.Slope) : x;
        }

        public Real BackwardActivate(Real gy, Real y)
        {
            return y <= 0 ? (Real)(y * this.Slope) : gy;
        }
    }
}
