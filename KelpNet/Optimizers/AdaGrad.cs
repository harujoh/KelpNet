using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaGrad : Optimizer
    {
        public Real LearningRate;
        public Real Epsilon;

        public AdaGrad(double learningRate = 0.01, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaGradParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdaGradParameter : OptimizerParameter
    {
        private readonly AdaGrad optimizer;
        private readonly Real[] h;

        public AdaGradParameter(FunctionParameter functionParameter, AdaGrad optimizer) : base(functionParameter)
        {
            this.h = new Real[functionParameter.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];

                this.h[i] += grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimizer.LearningRate * grad / (Math.Sqrt(this.h[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
