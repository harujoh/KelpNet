using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class Adam : Optimizer
    {
        public Real Alpha;
        public Real Beta1;
        public Real Beta2;
        public Real Epsilon;

        public Adam(Real alpha = 0.001f, Real beta1 = 0.9f, Real beta2 = 0.999f, Real epsilon = 1e-8f)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
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

            Real learningRate = this._optimizer.Alpha * (Real)Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Data[i] -= learningRate * this.m[i] / ((Real)Math.Sqrt(this.v[i]) + this._optimizer.Epsilon);
            }
        }
    }

}
