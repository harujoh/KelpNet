using System.Collections.Generic;
using System.Runtime.Serialization;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "LeakyReLU")]
    public class LeakyReLU : CPU.LeakyReLU, ICompressibleActivation
    {
        const string FUNCTION_NAME = "LeakyReLU";
        private const string PARAM_NAME = "/*slope*/";

        public ComputeKernel ForwardKernel { get; set; }
        public ComputeKernel BackwardKernel { get; set; }

        [DataMember]
        public string ActivateFunctionString { get; set; }

        [DataMember]
        public string ActivateKernelString { get; set; }

        [DataMember]
        public KeyValuePair<string, string>[] ActivationParameters { get; set; }

        [DataMember]
        public string ForwardKernelName { get; set; }

        [DataMember]
        public string BackwardKernelName { get; set; }

        [DataMember]
        public bool IsParallel { get; set; }

        void IParallelizable.InitParallel()
        {
            this.InitParallel();
        }

        bool IParallelizable.SetParallel(bool enable)
        {
            return this.SetParallel(enable);
        }

        public LeakyReLU(double slope = 0.2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(slope, name, inputNames, outputNames)
        {
            this.Initialize(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.LeakyReLU), new[] { new KeyValuePair<string, string>(PARAM_NAME, slope.ToString()) }, gpuEnable);
        }
    }
}
