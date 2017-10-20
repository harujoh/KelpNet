using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Mathmetrics.Trigonometric
{
    public class Tan : SingleInputFunction
    {
        private const string FUNCTION_NAME = "Tan";

        public Tan(string name = FUNCTION_NAME) : base(name)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = (Real)Math.Tan(x.Data[i]);
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = Math.Cos(x.Data[i]);
                x.Grad[i] += 1 / (gx * gx) * y.Grad[i];
            }
        }
    }
}
