using System;

namespace KelpNet
{
    [Serializable]
    public class Adam : Optimizer
    {
        public Real Alpha;
        public Real Beta1;
        public Real Beta2;
        public Real Epsilon;
        public Real Eta;
        public Real WeightDecayRate;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0, double weightDecayRate = 0)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
            this.Eta = eta;
            this.WeightDecayRate = weightDecayRate;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
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

        public AdamParameter(NdArray parameter, Adam optimizer) : base(parameter)
        {
            this.m = new Real[parameter.Data.Length];
            this.v = new Real[parameter.Data.Length];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Real fix1 = this._optimizer.Beta1;
            Real fix2 = this._optimizer.Beta2;

            for (int i = 1; i < this._optimizer.UpdateCount; i++)
            {
                fix1 *= this._optimizer.Beta1;
                fix2 *= this._optimizer.Beta2;
            }

            fix1 = 1 - fix1;
            fix2 = 1 - fix2;

            Real learningRate = this._optimizer.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * (learningRate * this.m[i] / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon)) + this._optimizer.WeightDecayRate * this.FunctionParameter.Data[i];
            }
        }
    }

}
