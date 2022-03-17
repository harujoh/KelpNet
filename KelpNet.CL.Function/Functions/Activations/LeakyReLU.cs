using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.Serialization;
using KelpNet.CL.Common;
using KelpNet.CL.Properties;

namespace KelpNet.CL
{
    [DataContract(Name = "LeakyReLU", Namespace = "KelpNet")]
    public class LeakyReLU<T> : CPU.LeakyReLU<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        public string FunctionName => "LeakyReLU";
        public string KernelSource { get; set; }

        private const string PARAM_NAME = "/*slope*/";

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
            KernelSource = OpenCL.GetKernelSource(Resources.LeakyReLU);
            bool result = this.SetParallel<T>(enable, new[] { new KeyValuePair<string, string>(PARAM_NAME, this.Slope.ToString()) });
            this.InitFunc(new StreamingContext());
            return result;
        }

        public LeakyReLU(T? slope = null, string name = "LeakyReLU", string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(slope, name, inputNames, outputNames)
        {
            this.SetParallel(gpuEnable, new[] { new KeyValuePair<string, string>(PARAM_NAME, this.Slope.ToString()) });
            this.InitFunc(new StreamingContext());
        }

        public LeakyReLU(CPU.LeakyReLU<T> leakyReLU) : base(leakyReLU.Name, leakyReLU.InputNames, leakyReLU.OutputNames)
        {
            this.Slope = leakyReLU.Slope;
            this.SetParallel(true, new[] { new KeyValuePair<string, string>(PARAM_NAME, this.Slope.ToString()) });
            this.InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            if (IsParallel)
            {
                switch (this)
                {
                    case LeakyReLU<float> leakyReLuF:
                        leakyReLuF.SingleInputForward = leakyReLuF.NeedPreviousForwardGpu;
                        leakyReLuF.SingleOutputBackward = leakyReLuF.NeedPreviousBackwardGpu;
                        break;

                    case LeakyReLU<double> leakyReLuD:
                        leakyReLuD.SingleInputForward = leakyReLuD.NeedPreviousForwardGpu;
                        leakyReLuD.SingleOutputBackward = leakyReLuD.NeedPreviousBackwardGpu;
                        break;
                }
            }
        }
    }
}
