using System;
using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    public abstract class Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public long UpdateCount = 0;

        public List<NdArray<T>> FunctionParameters = new List<NdArray<T>>();

        public List<Scheduler<T>> Schedulers = new List<Scheduler<T>>();

        public Action Update;
        public Action<int> UpdateFunctionParameters;

        public List<Action> ResetStates = new List<Action>();

        //個々の関数のパラメータ追加処理
        protected virtual void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
        }

        //個別にパラメータを追加する
        public void AddParameters(params NdArray<T>[] functionParameters)
        {
            FunctionParameters.AddRange(functionParameters);
            AddFunctionParameters(functionParameters);
        }

        public void SetUp(Function<T> function)
        {
            ResetStates.Add(function.ResetState);
            AddParameters(function.Parameters);
        }

        public virtual void Step()
        {
        }

        public void ResetState()
        {
            for (int i = 0; i < ResetStates.Count; i++)
            {
                ResetStates[i]();
            }
        }

        public void ResetParams()
        {
            for (int i = 0; i < this.FunctionParameters.Count; i++)
            {
                if (this.FunctionParameters[i].Grad != null)
                {
                    this.FunctionParameters[i].InitGrad();
                }

                //カウンタをリセット
                this.FunctionParameters[i].TrainCount = 0;
            }

            this.UpdateCount = 0;
        }
    }
}
