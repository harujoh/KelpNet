using System;
using System.Threading.Tasks;

namespace KelpNet
{
    [Serializable]
    public class AdaBound : Adam
    {
        private Real InitialAlpha;

        private Real Upper;
        private Real Lower;

        public double FinalLr;
        public double Gamma;

        public AdaBound(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double finalLr = 0.1,
            double gamma = 1e-3, double epsilon = 1e-8, double eta = 1.0) :
            base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            this.InitialAlpha = alpha;
            this.FinalLr = finalLr;
            this.Gamma = gamma;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaBoundParameter(functionParameter, this));
            }
        }

        public void UpdateBound()
        {
            FinalLr = FinalLr * Alpha / InitialAlpha;

            Lower = FinalLr * (1.0 - 1.0 / (Gamma * UpdateCount + 1));
            Upper = FinalLr * (1.0 + 1.0 / (Gamma * UpdateCount));
        }

        public Real Clip(Real val)
        {
            if (val <= Lower) return Lower;
            if (val >= Upper) return Upper;

            return val;
        }
    }

    [Serializable]
    public class AdaBoundParameter : OptimizerParameter
    {
        private readonly AdaBound _optimizer;

        private readonly Real[] m;
        private readonly Real[] v;

        public AdaBoundParameter(NdArray parameter, AdaBound optimizer) : base(parameter)
        {
            this.m = new Real[parameter.Data.Length];
            this.v = new Real[parameter.Data.Length];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Real alphaT = this._optimizer.AlphaT;
            _optimizer.UpdateBound();

            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                Real step = _optimizer.Clip(alphaT / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon));

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * step * this.m[i];
            });
        }
    }
}
