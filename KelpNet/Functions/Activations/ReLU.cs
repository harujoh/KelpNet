using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : NeedPreviousDataFunction
    {
        public ReLU(string name= "ReLU") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray y = NdArray.ZerosLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = Math.Max(0, x.Data[i]);
            }

            return y;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.ZerosLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return result;
        }
    }
}
