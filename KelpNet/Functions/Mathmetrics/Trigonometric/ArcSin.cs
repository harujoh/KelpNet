using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Mathmetrics.Trigonometric
{
    public class ArcSin : SingleInputFunction
    {
        private const string FUNCTION_NAME = "ArcSin";

        public ArcSin(string name = FUNCTION_NAME) : base(name)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = (Real)Math.Asin(x.Data[i]);
            }

            return new NdArray(resultData, x.Shape, x.BatchCount, this);
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += 1 / (Real)Math.Sqrt(-x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
