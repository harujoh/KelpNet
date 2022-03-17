using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "TanhActivation", Namespace = "KelpNet")]
    public class TanhActivation<T> : CPU.TanhActivation<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "TanhActivation";
        public string KernelSource { get; set; }

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
            KernelSource = OpenCL.GetKernelSource(Resources.TanhActivation);
            bool result = this.SetParallel<T>(enable);
            this.InitFunc(new StreamingContext());
            return result;
        }

        public TanhActivation(string name = "TanhActivation", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public TanhActivation(CPU.TanhActivation<T> tanhActivation) : base(tanhActivation.Name, tanhActivation.InputNames, tanhActivation.OutputNames)
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
                    case TanhActivation<float> tanhActivationF:
                        tanhActivationF.SingleInputForward = tanhActivationF.NeedPreviousForwardGpu;
                        tanhActivationF.SingleOutputBackward = tanhActivationF.NeedPreviousBackwardGpu;
                        break;

                    case TanhActivation<double> tanhActivationD:
                        tanhActivationD.SingleInputForward = tanhActivationD.NeedPreviousForwardGpu;
                        tanhActivationD.SingleOutputBackward = tanhActivationD.NeedPreviousBackwardGpu;
                        break;
                }
            }
        }
    }
}
