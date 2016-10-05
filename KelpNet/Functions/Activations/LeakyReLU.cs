using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousDataFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU") : base(name)
        {
            this._slope = slope;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray y = new NdArray(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (y.Data[i] < 0) y.Data[i] *= this._slope;
            }

            return y;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = new NdArray(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                if (prevOutput.Data[i] < 0)
                {
                    prevOutput.Data[i] *= this._slope;
                }
            }

            return result;
        }
    }
}
