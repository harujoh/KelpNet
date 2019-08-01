using System;

namespace KelpNet
{
    [Serializable]
    public class ELU : SingleInputFunction
    {
        const string FUNCTION_NAME = "ELU";

        public Real Alpha;

        public ELU(double alpha = 1, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Alpha = alpha;
        }

        public override NdArray SingleInputForward(NdArray x)
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
                    result[i] = this.Alpha * (Math.Exp(x.Data[i]) - 1);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    x.Grad[i] += y.Grad[i];
                }
                else
                {
                    x.Grad[i] += y.Grad[i] * this.Alpha * Math.Exp(x.Data[i]);
                }
            }
        }
    }
}
