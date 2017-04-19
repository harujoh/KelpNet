using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class RMSprop : Optimizer
    {
        public Real LearningRate;
        public Real Alpha;
        public Real Epsilon;

        public RMSprop(Real? learningRate = null, Real? alpha = null, Real? epsilon = null)
        {
            this.LearningRate = learningRate ?? 0.01f;
            this.Alpha = alpha ?? 0.99f;
            this.Epsilon = epsilon ?? (Real)1e-8f;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new RMSpropParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class RMSpropParameter : OptimizerParameter
    {
        private readonly RMSprop optimiser;
        private readonly Real[] ms;

        public RMSpropParameter(FunctionParameter parameter, RMSprop optimiser) : base(parameter)
        {
            this.optimiser = optimiser;
            this.ms = new Real[parameter.Length];
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];
                this.ms[i] *= this.optimiser.Alpha;
                this.ms[i] += (1 - this.optimiser.Alpha) * grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimiser.LearningRate * grad / ((Real)Math.Sqrt(this.ms[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
