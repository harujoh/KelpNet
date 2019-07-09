using System;
using System.Diagnostics;
using Cloo;

namespace KelpNet.CL
{
    [Serializable]
    public abstract class CompressibleFunction : KelpNet.CompressibleFunction
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgWKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgXKernel;

        public string ForwardKernelName { get; }
        public string BackwardgWKernelName { get; }
        public string BackwardgXKernelName { get; }

        protected string KernelString;

        protected abstract NdArray NeedPreviousForwardGpu(NdArray input);
        protected abstract void NeedPreviousBackwardGpu(NdArray y, NdArray x);

        protected CompressibleFunction(string functionName,string kernelString, KelpNet.CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(activation, name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardgWKernelName = kernelNameBase + "gWBackward";
            this.BackwardgXKernelName = kernelNameBase + "gXBackward";

            this.Activator = activation;

            this.KernelString = kernelString;

            this.SetParallel(gpuEnable);
        }

        public override bool SetParallel(bool enable)
        {
            this.IsParallel = enable & OpenCL.Enable;

            if (this.IsParallel)
            {
                this.InitParallel();

                this.SingleInputForward = this.NeedPreviousForwardGpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardGpu;
            }
            else
            {
                this.SingleInputForward = this.NeedPreviousForwardCpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
            }

            return IsParallel;
        }

        public override void InitParallel()
        {
            if (this.IsParallel)
            {
                string kernelSource = this.KernelString;

                if (this.Activator is CompressibleActivation activator)
                {
                    string activationSource = activator.ActivateFunctionString;

                    foreach (var activationParameter in activator.ActivationParameters)
                    {
                        activationSource = activationSource.Replace(activationParameter.Key, activationParameter.Value);
                    }

                    //アクティベーションを活性化
                    kernelSource = activationSource + kernelSource.Replace("/*ForwardActivate*/", "ForwardActivate");
                }

                ComputeProgram program = OpenCL.CreateProgram(kernelSource);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardgWKernel = program.CreateKernel(this.BackwardgWKernelName);
                this.BackwardgXKernel = program.CreateKernel(this.BackwardgXKernelName);
            }
        }
    }
}
