using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public abstract class CompressibleFunction<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation<T> Activator { get; protected set; }

        private readonly KeyValuePair<string, string>[] _activationParameters;

        protected abstract NdArray<T> NeedPreviousForwardCpu(NdArray<T> input);
        protected abstract void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x);

        protected CompressibleFunction(string functionName, CompressibleActivation<T> activation = null, KeyValuePair<string, string>[] activationParameters = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._activationParameters = activationParameters;

            this.SetActivation(activation);

            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation<T> activation)
        {
            this.Activator = activation;
        }
    }
}
