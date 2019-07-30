using System;
using System.Threading.Tasks;

namespace KelpNet
{
    [Serializable]
    public class AmsGrad : Adam
    {
        public AmsGrad(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0) :
            base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        { }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AmsGradParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    public class AmsGradParameter : OptimizerParameter
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
            Real alphaT = this._optimizer.AlphaT;

            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                if (this.vhat[i] < this.v[i])
                {
                    this.vhat[i] = this.v[i];
                }

                Real step = alphaT / (Math.Sqrt(this.vhat[i]) + this._optimizer.Epsilon);

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * step * this.m[i];
            });
        }
    }
}
