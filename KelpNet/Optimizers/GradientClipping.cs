using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping : Optimizer
    {
        public Real Threshold;

        public GradientClipping(Real threshold)
        {
            this.Threshold = threshold;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new GradientClippingParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class GradientClippingParameter : OptimizerParameter
    {
        private readonly GradientClipping optimizer;

        public GradientClippingParameter(NdArray functionParameter, GradientClipping optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            //_sum_sqnorm
            Real s = 0;

            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                s += this.FunctionParameter.Grad[i] * this.FunctionParameter.Grad[i];
            }

            Real norm = (Real)Math.Sqrt(s);
            Real rate = this.optimizer.Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
                {
                    this.FunctionParameter.Grad[i] *= rate;
                }
            }
        }
    }
}
