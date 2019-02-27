using System;

namespace KelpNet
{
    [Serializable]
    public class Swish : SingleInputFunction
    {
        const string FUNCTION_NAME = "Swish";

        private Real _beta;

        public Swish(double beta = 1.0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _beta = beta;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {                
                result[i] = x.Data[i] * (Math.Tanh(x.Data[i] * _beta * 0.5) * 0.5 + 0.5);
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real sig = Math.Tanh(_beta * x.Data[i] * 0.5) * 0.5 + 0.5;
                Real by = _beta * x.Data[i] * sig;

                x.Grad[i] += y.Grad[i] * (by + sig * (1 - by));
            }
        }
    }
}
