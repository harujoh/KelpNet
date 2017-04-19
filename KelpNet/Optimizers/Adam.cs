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

        public Adam(Real? alpha = null, Real? beta1 = null, Real? beta2 = null, Real? epsilon = null)
        {
            this.Alpha = alpha ?? 0.001f;
            this.Beta1 = beta1 ?? 0.9f;
            this.Beta2 = beta2 ?? 0.999f;
            this.Epsilon = epsilon ??(Real)1e-8f;
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
        private readonly Adam optimiser;

        private readonly Real[] m;
        private readonly Real[] v;

        public AdamParameter(FunctionParameter parameter, Adam optimiser) : base(parameter)
        {
            this.m = new Real[parameter.Length];
            this.v = new Real[parameter.Length];

            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            Real fix1 = 1 - (Real)Math.Pow(this.optimiser.Beta1, this.optimiser.UpdateCount);
            Real fix2 = 1 - (Real)Math.Pow(this.optimiser.Beta2, this.optimiser.UpdateCount);
            Real lr = this.optimiser.Alpha * (Real)Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];

                this.m[i] += (1 - this.optimiser.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this.optimiser.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Param.Data[i] -= lr * this.m[i] / ((Real)Math.Sqrt(this.v[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
