using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "ReLU", Namespace = "KelpNet")]
    public class ReLU<T> : CPU.ReLU<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "ReLU";
        public string KernelSource { get; set; } = OpenCL.GetKernelSource(Resources.ReLU);

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel { get; set; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel { get; set; }

        [DataMember]
        public string ActivateKernelString { get; set; }

        [DataMember]
        public string ForwardKernelName { get; set; }

        [DataMember]
        public string BackwardKernelName { get; set; }

        [DataMember]
        public bool IsParallel { get; set; }


        public bool SetParallel(bool enable)
        {
            bool result = this.SetParallel<T>(enable);
            this.InitFunc(new StreamingContext());
            return result;
        }

        public ReLU(string name = "ReLU", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public ReLU(CPU.ReLU<T> reLU) : base(reLU.Name, reLU.InputNames, reLU.OutputNames)
        {
            this.SetParallel(true);
            this.InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case ReLU<float> ReLUF:
                        ReLUF.SingleInputForward = ReLUF.NeedPreviousForwardGpu;
                        ReLUF.SingleOutputBackward = ReLUF.NeedPreviousBackwardGpu;
                        break;

                    case ReLU<double> ReLUD:
                        ReLUD.SingleInputForward = ReLUD.NeedPreviousForwardGpu;
                        ReLUD.SingleOutputBackward = ReLUD.NeedPreviousBackwardGpu;
                        break;
                }
            }
        }
    }
}
