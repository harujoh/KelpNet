using System;
using System.Collections.Generic;
using Cloo;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleFunction : NeedPreviousInputFunction, IParallelizable
    {
        public CompressibleActivation Activation { get; protected set; }

        [NonSerialized]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        public ComputeKernel BackwardgWKernel;

        [NonSerialized]
        public ComputeKernel BackwardgXKernel;

        public string ForwardKernelName { get; }
        public string BackwardgWKernelName { get; }
        public string BackwardgXKernelName { get; }

        protected string KernelString;

        private KeyValuePair<string, string>[] _activationParameters;

        protected abstract BatchArray NeedPreviousForwardCpu(BatchArray input);
        protected abstract BatchArray NeedPreviousForwardGpu(BatchArray input);
        protected abstract BatchArray NeedPreviousBackwardCpu(BatchArray gy, BatchArray x);
        protected abstract BatchArray NeedPreviousBackwardGpu(BatchArray gy, BatchArray x);

        protected CompressibleFunction(string name, int inputCount, int oututCount, bool gpuEnable, string functionName, CompressibleActivation activation = null, params KeyValuePair<string, string>[] activationParameters) : base(name, inputCount, oututCount)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardgWKernelName = kernelNameBase + "gWBackward";
            this.BackwardgXKernelName = kernelNameBase + "gXBackward";

            this.KernelString = Weaver.GetKernelSource(functionName);

            _activationParameters = activationParameters;

            this.SetActivation(activation);

            this.SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            if (GpuEnable)
            {
                CreateKernel();
                NeedPreviousForward = NeedPreviousForwardGpu;
                NeedPreviousBackward = NeedPreviousBackwardGpu;
            }
            else
            {
                NeedPreviousForward = NeedPreviousForwardCpu;
                NeedPreviousBackward = NeedPreviousBackwardCpu;
            }

            return GpuEnable;
        }

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation activation)
        {
            this.Activation = activation;

            if (this.Activation != null)
            {
                foreach (var activationParameterer in _activationParameters)
                {
                    KernelString = KernelString.Replace(activationParameterer.Key, activationParameterer.Value);
                }
            }

            //Kernelの再構築が必要
            if (this.GpuEnable)
            {
                CreateKernel();
            }
        }

        public void CreateKernel()
        {
            string kernelSource = KernelString;

            if (this.Activation != null)
            {
                kernelSource = this.Activation.ActivateFunctionString + KernelString;
            }

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
            BackwardgWKernel = program.CreateKernel(BackwardgWKernelName);
            BackwardgXKernel = program.CreateKernel(BackwardgXKernelName);
        }
    }
}
