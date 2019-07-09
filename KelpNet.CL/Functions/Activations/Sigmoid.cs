using System;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [Serializable]
    public class Sigmoid : CompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Sigmoid), null, name, inputNames, outputNames, gpuEnable)
        {
        }

        public override Real ForwardActivate(Real x)
        {
            return Math.Tanh(x * 0.5) * 0.5 + 0.5;
        }

        public override Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0 - y);
        }
    }
}
