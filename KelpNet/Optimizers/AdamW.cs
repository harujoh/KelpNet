using System;
using System.Threading.Tasks;

namespace KelpNet
{
    [Serializable]
    public class AdamW : Adam
    {
        public Real WeightDecayRate;

        public AdamW(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0, double weightDecayRate = 0) :
            base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            this.WeightDecayRate = weightDecayRate;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamWParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    public class AdamWParameter : OptimizerParameter
    {
        private readonly AdamW _optimizer;

        private readonly Real[] m;
        private readonly Real[] v;

        public AdamWParameter(NdArray parameter, AdamW optimizer) : base(parameter)
        {
            this.m = new Real[parameter.Data.Length];
            this.v = new Real[parameter.Data.Length];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Real alphaT = this._optimizer.AlphaT;

            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                Real step = alphaT / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon);

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * (step * this.m[i] + this._optimizer.WeightDecayRate * this.FunctionParameter.Data[i]);
            });
        }
    }
}
