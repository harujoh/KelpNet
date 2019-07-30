using System;
using System.Threading.Tasks;

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

        public Real AlphaT
        {
            get
            {
                Real fix1 = 1 - Math.Pow(Beta1, UpdateCount);
                Real fix2 = 1 - Math.Pow(Beta2, UpdateCount);

                return Alpha * Math.Sqrt(fix2) / fix1;
            }
        }

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
            this.Eta = eta;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamParameter(functionParameter, this));
            }
        }

        public override void Step()
        {
            for (int i = 0; i < Schedulers.Count; i++)
            {
                Alpha = Schedulers[i].Step(Alpha);
            }
        }
    }

    [Serializable]
    public class AdamParameter : OptimizerParameter
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
            Real alphaT = this._optimizer.AlphaT;

            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                Real step = alphaT / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon);

                this.FunctionParameter.Data[i] -= this._optimizer.Eta * step * this.m[i];
            });
        }
    }
}
