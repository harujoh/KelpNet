using System;

namespace KelpNet
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation Activator { get; protected set; }

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);
        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);

        public bool IsParallel { get; set; }

        protected CompressibleFunction(CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Activator = activation;

            this.IsParallel = false;
            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        public bool SetParallel(bool enable)
        {
            return false;
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
        }
    }
}
