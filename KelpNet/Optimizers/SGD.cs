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
    public class SGD : Optimizer
    {
        public Real LearningRate;

        public SGD(Real learningRate = 0.1f)
        {
            this.LearningRate = learningRate;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new SGDParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class SGDParameter : OptimizerParameter
    {
        private readonly SGD optimizer;

        public SGDParameter(NdArray functionParameter, SGD optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad[i];
            }
        }
    }

}
