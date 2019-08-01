using System;

namespace KelpNet
{
    public class ArcTan : SingleInputFunction
    {
        private const string FUNCTION_NAME = "ArcTan";

        public ArcTan(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = (Real)Math.Atan(x.Data[i]);
            }

            return new NdArray(resultData, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += 1 / (x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
