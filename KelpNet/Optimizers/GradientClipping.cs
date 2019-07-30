using System;
using System.Threading.Tasks;

namespace KelpNet
{
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping : Optimizer
    {
        public Real Threshold;

        public GradientClipping(double threshold)
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
    public class GradientClippingParameter : OptimizerParameter
    {
        private readonly GradientClipping optimizer;

        public GradientClippingParameter(NdArray functionParameter, GradientClipping optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            //_sum_sqnorm
            double s = 0;

            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                s += this.FunctionParameter.Grad[i] * this.FunctionParameter.Grad[i];
            }

            double norm = Math.Sqrt(s);
            double rate = this.optimizer.Threshold / norm;

            if (rate < 1)
            {
                Parallel.For(0, FunctionParameter.Data.Length, i =>
                {
                    this.FunctionParameter.Grad[i] *= rate;
                });
            }
        }
    }
}
