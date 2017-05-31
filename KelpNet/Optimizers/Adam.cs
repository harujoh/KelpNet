using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class Adam : Optimizer
    {
        public Real Alpha;
        public Real Beta1;
        public Real Beta2;
        public Real Epsilon;

        public double LearningRate
        {
            get
            {
                double fix1 = 1 - Math.Pow(this.Beta1, UpdateCount);
                double fix2 = 1 - Math.Pow(this.Beta2, UpdateCount);
                return this.Alpha * Math.Sqrt(fix2) / fix1;
            }
        }

        public Adam(Real? alpha = null, Real? beta1 = null, Real? beta2 = null, Real? epsilon = null)
        {
            this.Alpha = alpha ?? (Real)0.001;
            this.Beta1 = beta1 ?? (Real)0.9;
            this.Beta2 = beta2 ?? (Real)0.999;
            this.Epsilon = epsilon ?? (Real)1e-8;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdamParameter : OptimizerParameter
    {
        private readonly Adam _optimizer;

        private readonly Real[] m;
        private readonly Real[] v;

        public AdamParameter(FunctionParameter parameter, Adam optimizer) : base(parameter)
        {
            this.m = new Real[parameter.Length];
            this.v = new Real[parameter.Length];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Param.Data[i] -= this._optimizer.LearningRate * this.m[i] / ((Real)Math.Sqrt(this.v[i]) + this._optimizer.Epsilon);
            }
        }
    }

}
