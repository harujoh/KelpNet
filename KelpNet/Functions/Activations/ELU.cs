using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        const string FUNCTION_NAME = "ELU";

        private readonly Real _alpha;

        public ELU(double alpha = 1, string name = FUNCTION_NAME) : base(name)
        {
            this._alpha = alpha;

            NeedPreviousForward = NeedPreviousForwardCpu;
            NeedPreviousBackward = NeedPreviousBackwardCpu;
        }

        protected BatchArray NeedPreviousForwardCpu(BatchArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    result[i] = x.Data[i];
                }
                else
                {
                    result[i] = this._alpha * (Math.Exp(x.Data[i]) - 1);
                }
            }

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        protected BatchArray NeedPreviousBackwardCpu(BatchArray gy, BatchArray prevInput, BatchArray prevOutput)
        {
            Real[] result = new Real[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                if (prevOutput.Data[i] >= 0)
                {
                    result[i] = gy.Data[i];
                }
                else
                {
                    result[i] = gy.Data[i] * this._alpha * Math.Exp(prevInput.Data[i]);
                }
            }

            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }
    }
}
