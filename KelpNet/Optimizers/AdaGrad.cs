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
    public class AdaGrad : Optimizer
    {
        public Real LearningRate;
        public Real Epsilon;

        public AdaGrad(Real learningRate = 0.01f, Real epsilon = 1e-8f)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
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

        public AdaGradParameter(NdArray functionParameter, AdaGrad optimizer) : base(functionParameter)
        {
            this.h = new Real[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];

                this.h[i] += grad * grad;

                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * grad / (Real)(Math.Sqrt(this.h[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
