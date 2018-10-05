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
    public class RMSprop : Optimizer
    {
        public Real LearningRate;
        public Real Alpha;
        public Real Epsilon;

        public RMSprop(Real learningRate = 0.01f, Real alpha = 0.99f, Real epsilon = 1e-8f)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new RMSpropParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class RMSpropParameter : OptimizerParameter
    {
        private readonly RMSprop optimizer;
        private readonly Real[] ms;

        public RMSpropParameter(NdArray parameter, RMSprop optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            this.ms = new Real[parameter.Data.Length];
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];
                this.ms[i] *= this.optimizer.Alpha;
                this.ms[i] += (1 - this.optimizer.Alpha) * grad * grad;

                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * grad / ((Real)Math.Sqrt(this.ms[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
