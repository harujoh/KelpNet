using System.Collections.Generic;
using System.Runtime.Serialization;
using Cloo;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "Sigmoid")]
    public class Sigmoid : CPU.Sigmoid, ICompressibleActivation
    {
        const string FUNCTION_NAME = "Sigmoid";

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

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.Initialize(FUNCTION_NAME, OpenCL.GetKernelSource(Resources.Sigmoid), null, gpuEnable);
        }
    }
}
