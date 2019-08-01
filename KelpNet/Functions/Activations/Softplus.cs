using System;

namespace KelpNet
{
    public class Softplus : SingleInputFunction
    {
        const string FUNCTION_NAME = "Softplus";

        public Real Beta;
        public Real BetaInv;

        public Softplus(double beta = 1, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Beta = beta;
            this.BetaInv = 1 / this.Beta;
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * this.Beta;
                }

                Real maxval = y[b * x.Length];
                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * this.Beta)))) * this.BetaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (1 - 1 / (1 + Math.Exp(this.Beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

