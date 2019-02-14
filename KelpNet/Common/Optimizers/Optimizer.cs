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

        internal abstract void AddFunctionParameters(NdArray[] functionParameters);

        public void Update()
        {
            bool isUpdated = false;

            for (int i = 0; i < this.OptimizerParameters.Count; i++)
            {
                //傾きの割引を実行して更新があったかチェックをする
                if (this.OptimizerParameters[i].FunctionParameter.Reduce())
                {
                    this.OptimizerParameters[i].UpdateFunctionParameters();

                    this.OptimizerParameters[i].FunctionParameter.ClearGrad();
                    //カウンタをリセット
                    this.OptimizerParameters[i].FunctionParameter.TrainCount = 0;

                    isUpdated = true;
                }
            }

            if (isUpdated)
            {
                this.UpdateCount++;
            }
        }

        public void ResetParams()
        {
            for (int i = 0; i < this.OptimizerParameters.Count; i++)
            {
                this.OptimizerParameters[i].FunctionParameter.ClearGrad();
                //カウンタをリセット
                this.OptimizerParameters[i].FunctionParameter.TrainCount = 0;
            }

            this.UpdateCount = 0;
        }
    }

    //このクラスはFunctionParameterと1:1で作成される
    [Serializable]
    public abstract class OptimizerParameter
    {
        public NdArray FunctionParameter;

        protected OptimizerParameter(NdArray functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public abstract void UpdateFunctionParameters();
    }
}
