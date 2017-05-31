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
            this.LearningRate = learningRate ?? (Real)0.01;
            this.Alpha = alpha ?? (Real)0.99;
            this.Epsilon = epsilon ?? (Real)1e-8;
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
        private readonly RMSprop optimizer;
        private readonly Real[] ms;

        public RMSpropParameter(FunctionParameter parameter, RMSprop optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            this.ms = new Real[parameter.Length];
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];
                this.ms[i] *= this.optimizer.Alpha;
                this.ms[i] += (1 - this.optimizer.Alpha) * grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimizer.LearningRate * grad / ((Real)Math.Sqrt(this.ms[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
