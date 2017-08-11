using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
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

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new GradientClippingParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class GradientClippingParameter : OptimizerParameter
    {
        private readonly GradientClipping optimizer;

        public GradientClippingParameter(FunctionParameter functionParameter, GradientClipping optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            //_sum_sqnorm
            double s = 0;

            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                s += Math.Pow(this.FunctionParameter.Grad.Data[i], 2);
            }

            double norm = Math.Sqrt(s);
            double rate = this.optimizer.Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < this.FunctionParameter.Length; i++)
                {
                    this.FunctionParameter.Grad.Data[i] *= rate;
                }
            }
        }
    }
}
