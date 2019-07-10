using System;

namespace KelpNet
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction, ICompressibleFunction
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public ICompressibleActivation Activator { get; set; }

        public abstract NdArray NeedPreviousForwardCpu(NdArray input);
        public abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);

        protected CompressibleFunction(ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Activator = activation;

            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        //後からActivationを追加する用
        public virtual void SetActivation(CompressibleActivation activation)
        {
            this.Activator = activation;
        }
    }
}
