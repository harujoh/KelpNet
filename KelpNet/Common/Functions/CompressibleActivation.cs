using System;

namespace KelpNet
{
    [Serializable]
    public abstract class CompressibleActivation : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Activation";

        //.Netで使用するActivateの仮想関数
        public abstract Real ForwardActivate(Real x);
        public abstract Real BackwardActivate(Real gy, Real y);

        public bool IsParallel { get; set; }

        protected CompressibleActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.IsParallel = false;
            this.SetParallel(gpuEnable);

            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        public bool SetParallel(bool enable)
        {
            return false;
        }

        public void InitParallel()
        {
        }

        private NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = this.ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += this.BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }
    }
}
