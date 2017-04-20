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

        public AdaGrad(Real? learningRate = null, Real? epsilon = null)
        {
            this.LearningRate = learningRate ?? (Real)0.01;
            this.Epsilon = epsilon ?? (Real)1e-8;
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
        private readonly AdaGrad optimiser;
        private readonly Real[] h;

        public AdaGradParameter(FunctionParameter functionParameter, AdaGrad optimiser) : base(functionParameter)
        {
            this.h = new Real[functionParameter.Length];
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];

                this.h[i] += grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimiser.LearningRate * grad / ((Real)Math.Sqrt(this.h[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
