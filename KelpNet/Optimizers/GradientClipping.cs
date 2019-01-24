using System;

namespace KelpNet
{
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> Threshold;

        public GradientClipping(Real<T> threshold)
        {
            this.Threshold = threshold;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new GradientClippingParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class GradientClippingParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly GradientClipping<T> optimizer;

        public GradientClippingParameter(NdArray<T> functionParameter, GradientClipping<T> optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            //_sum_sqnorm
            Real<T> s = 0;

            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                s += this.FunctionParameter.Grad[i] * this.FunctionParameter.Grad[i];
            }

            Real<T> norm = (Real<T>)Math.Sqrt(s);
            Real<T> rate = this.optimizer.Threshold / norm;

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
