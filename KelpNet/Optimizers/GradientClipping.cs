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

        public GradientClipping(Real threshold)
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
        private readonly GradientClipping optimiser;

        public GradientClippingParameter(FunctionParameter functionParameter, GradientClipping optimiser) : base(functionParameter)
        {
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            //_sum_sqnorm
            Real s = 0.0f;

            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                s += (Real)Math.Pow(this.FunctionParameter.Grad.Data[i], 2);
            }

            Real norm = (Real)Math.Sqrt(s);
            Real rate = this.optimiser.Threshold / norm;

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
