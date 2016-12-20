using System;
using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    [Serializable]
    public abstract class Optimizer
    {
        public long UpdateCount = 1;
        protected List<OptimizerParameter> OptimizerParameters = new List<OptimizerParameter>();

        internal abstract void AddFunctionParameters(FunctionParameter[] functionParameters);

        public void Update()
        {
            for (int i = 0; i < this.OptimizerParameters.Count; i++)
            {
                this.OptimizerParameters[i].UpdateFunctionParameters();
            }

            this.UpdateCount++;
        }
    }

    //このクラスはFunctionParameterと1:1で作成される
    [Serializable]
    public abstract class OptimizerParameter
    {
        protected FunctionParameter FunctionParameter;

        protected OptimizerParameter(FunctionParameter functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public abstract void UpdateFunctionParameters();
    }
}
