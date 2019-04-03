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
        public bool IsAmsGrad;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0, double weightDecayRate = 0, bool isAmsGrad = false)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
            this.Eta = eta;
            this.WeightDecayRate = weightDecayRate;
            this.IsAmsGrad = isAmsGrad;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            if (this.IsAmsGrad)
            {
                foreach (NdArray functionParameter in functionParameters)
                {
                    this.OptimizerParameters.Add(new AmsGradParameter(functionParameter, this));
                }
            }
            else
            {
                foreach (NdArray functionParameter in functionParameters)
                {
                    this.OptimizerParameters.Add(new AdamParameter(functionParameter, this));
                }
            }
        }
    }

    [Serializable]
    class AmsGradParameter : OptimizerParameter
    {
        private readonly Adam _optimizer;

        private readonly Real[] m;
        private readonly Real[] v;
        private readonly Real[] vhat;

        public AmsGradParameter(NdArray parameter, Adam optimizer) : base(parameter)
        {
            this.m = new Real[parameter.Data.Length];
            this.v = new Real[parameter.Data.Length];
            this.vhat = new Real[parameter.Data.Length];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Real fix1 = 1 - Math.Pow(this._optimizer.Beta1, this._optimizer.UpdateCount);
            Real fix2 = 1 - Math.Pow(this._optimizer.Beta2, this._optimizer.UpdateCount);

            Real learningRate = this._optimizer.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                if (this.vhat[i] < this.v[i])
                {
                    this.vhat[i] = this.v[i];
                }

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * (learningRate * this.m[i] / (Math.Sqrt(this.vhat[i]) + this._optimizer.Epsilon) + this._optimizer.WeightDecayRate * this.FunctionParameter.Data[i]);
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
            Real fix1 = 1 - Math.Pow(this._optimizer.Beta1, this._optimizer.UpdateCount);
            Real fix2 = 1 - Math.Pow(this._optimizer.Beta2, this._optimizer.UpdateCount);

            Real learningRate = this._optimizer.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * (learningRate * this.m[i] / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon) + this._optimizer.WeightDecayRate * this.FunctionParameter.Data[i]);
            }
        }
    }

}
