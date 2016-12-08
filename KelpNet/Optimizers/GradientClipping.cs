using System;

namespace KelpNet.Optimizers
{
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping : Optimizer
    {
        public double Threshold;

        public GradientClipping(double threshold)
        {
            this.Threshold = threshold;
        }

        public override void Initilise(OptimizeParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new GradientClippingParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class GradientClippingParameter : OptimizerParameter
    {
        private readonly GradientClipping optimiser;

        public GradientClippingParameter(OptimizeParameter functionParameter, GradientClipping optimiser) : base(functionParameter)
        {
            this.optimiser = optimiser;
        }

        public override void Update()
        {
            //_sum_sqnorm
            double s = 0.0;

            for (int i = 0; i < this.FunctionParameters.Length; i++)
            {
                s += Math.Pow(this.FunctionParameters.Grad.Data[i], 2);
            }

            double norm = Math.Sqrt(s);
            double rate = this.optimiser.Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < this.FunctionParameters.Length; i++)
                {
                    this.FunctionParameters.Grad.Data[i] *= rate;
                }
            }
        }
    }
}
