using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Swish : SingleInputFunction
    {
        const string FUNCTION_NAME = "Swish";

        public Swish(string name = FUNCTION_NAME) : base(name)
        {
        }

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] / (1 + Math.Exp(-x.Data[i]));
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * (y.Data[i] + y.Data[i] / x.Data[i] * (1 - y.Data[i]));
            }
        }
    }
}
