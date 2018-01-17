using System;
using System.Collections.Generic;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleActivation : SingleInputFunction
    {
        const string FUNCTION_NAME = "Activation";

        //GPU向けのActivate関数の文字列
        public string ActivateFunctionString;

        //.Netで使用するActivateの仮想関数
        internal abstract Real ForwardActivate(Real x);
        internal abstract Real BackwardActivate(Real gy, Real y);

        protected string ActivateKernelString;

        protected CompressibleActivation(string functionName, KeyValuePair<string, string>[] parameters, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
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
