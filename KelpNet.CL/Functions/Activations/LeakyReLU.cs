using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "LeakyReLU", Namespace = "KelpNet")]
    public class LeakyReLU : CPU.LeakyReLU, ICompressibleActivation
    {
        public string FunctionName => "LeakyReLU";
        public string KernelSource => OpenCL.GetKernelSource(Resources.LeakyReLU);

        private const string PARAM_NAME = "/*slope*/";

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel { get; set; }

        [DataMember]
        public KeyValuePair<string, string>[] ActivationParameters { get; set; }

        [DataMember]
        public string ActivateKernelString { get; set; }

        [DataMember]
        public string ForwardKernelName { get; set; }

        [DataMember]
        public string BackwardKernelName { get; set; }

        [DataMember]
        public bool IsParallel { get; set; }


        bool IParallelizable.SetParallel(bool enable)
        {
            return this.SetParallel(enable);
        }

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(slope, name, inputNames, outputNames)
        {
            this.InitParam();
            this.SetParallel(gpuEnable);
        }

        public LeakyReLU(CPU.LeakyReLU leakyReLU) : base(leakyReLU.Name, leakyReLU.InputNames, leakyReLU.OutputNames)
        {
            this.Slope = leakyReLU.Slope;

            this.InitParam();
            this.SetParallel(true);
        }

        public void InitParam()
        {
            this.ActivationParameters = new[] { new KeyValuePair<string, string>(PARAM_NAME, this.Slope.ToString()) };
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            return IsParallel ? this.NeedPreviousForwardGpu(x) : base.SingleInputForward(x);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            if (IsParallel)
            {
                this.NeedPreviousBackwardGpu(y, x);
            }
            else
            {
                base.SingleOutputBackward(y, x);
            }
        }
    }
}
