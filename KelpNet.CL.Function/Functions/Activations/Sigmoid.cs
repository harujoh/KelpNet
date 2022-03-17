using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "Sigmoid", Namespace = "KelpNet")]
    public class Sigmoid<T> : CPU.Sigmoid<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "Sigmoid";
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
            KernelSource = OpenCL.GetKernelSource(Resources.Sigmoid);
            bool result =  this.SetParallel<T>(enable);
            this.InitFunc(new StreamingContext());
            return result;
        }

        public Sigmoid(string name = "Sigmoid", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable);
            this.InitFunc(new StreamingContext());
        }

        public Sigmoid(CPU.Sigmoid<T> sigmoid) : base(sigmoid.Name, sigmoid.InputNames, sigmoid.OutputNames)
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
                    case Sigmoid<float> sigmoidF:
                        sigmoidF.SingleInputForward = sigmoidF.NeedPreviousForwardGpu;
                        sigmoidF.SingleOutputBackward = sigmoidF.NeedPreviousBackwardGpu;
                        break;

                    case Sigmoid<double> sigmoidD:
                        sigmoidD.SingleInputForward = sigmoidD.NeedPreviousForwardGpu;
                        sigmoidD.SingleOutputBackward = sigmoidD.NeedPreviousBackwardGpu;
                        break;
                }
            }
        }

    }
}
