using System;
using System.Collections.Generic;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation Activator { get; protected set; }

        protected string KernelString;

        private readonly KeyValuePair<string, string>[] _activationParameters;

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);
        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);

        protected CompressibleFunction(string functionName, CompressibleActivation activation = null, KeyValuePair<string, string>[] activationParameters = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            _activationParameters = activationParameters;

            this.SetActivation(activation);

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation activation)
        {
            this.Activator = activation;

            if (this.Activator != null)
            {
                foreach (var activationParameterer in _activationParameters)
                {
                    KernelString = KernelString.Replace(activationParameterer.Key, activationParameterer.Value);
                }
            }
        }
    }
}
