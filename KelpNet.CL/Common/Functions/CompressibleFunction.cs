using System;
using System.Diagnostics;
using Cloo;

namespace KelpNet.CL
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation Activator { get; protected set; }

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

        protected abstract string KernelString { get; }

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);
        protected abstract NdArray NeedPreviousForwardGpu(NdArray input);
        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);
        protected abstract void NeedPreviousBackwardGpu(NdArray y, NdArray x);

        public bool IsParallel { get; set; }

        protected CompressibleFunction(string functionName, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardgWKernelName = kernelNameBase + "gWBackward";
            this.BackwardgXKernelName = kernelNameBase + "gXBackward";

            this.Activator = activation;

            this.SetParallel(gpuEnable);
        }

        public bool SetParallel(bool enable)
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

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation activation)
        {
            this.Activator = activation;

            //Kernelの再構築が必要
            InitParallel();
        }

        public void InitParallel()
        {
            if (this.IsParallel)
            {
                string kernelSource = this.KernelString;

                if (this.Activator != null)
                {
                    string activationSource = this.Activator.ActivateFunctionString;

                    foreach (var activationParameter in this.Activator.ActivationParameters)
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
