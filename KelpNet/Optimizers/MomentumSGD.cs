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
    public class MomentumSGD : Optimizer
    {
        public Real LearningRate;
        public Real Momentum;

        public MomentumSGD(Real learningRate = 0.01f, Real momentum = 0.9f)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new MomentumSGDParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class MomentumSGDParameter : OptimizerParameter
    {
        private readonly MomentumSGD optimizer;
        private readonly Real[] v;

        public MomentumSGDParameter(NdArray functionParameter, MomentumSGD optimizer) : base(functionParameter)
        {
            this.v = new Real[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                this.v[i] *= this.optimizer.Momentum;
                this.v[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad[i];

                this.FunctionParameter.Data[i] += this.v[i];
            }
        }
    }

}
