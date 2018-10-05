using System;
using System.Collections.Generic;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation Activator { get; protected set; }

        private readonly KeyValuePair<string, string>[] _activationParameters;

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);
        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);

        protected CompressibleFunction(string functionName, CompressibleActivation activation = null, KeyValuePair<string, string>[] activationParameters = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._activationParameters = activationParameters;

            this.SetActivation(activation);

            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation activation)
        {
            this.Activator = activation;
        }
    }
}
