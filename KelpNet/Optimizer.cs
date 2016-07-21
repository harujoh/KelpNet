using System.Collections.Generic;
using KelpNet.Functions;

namespace KelpNet
{
    //Optimizerの基底クラス
    public abstract class Optimizer
    {
        //更新回数のカウント
        protected double t = 1;

        public void Update(List<OptimizableFunction> optimizableFunctions)
        {
            this.Update(optimizableFunctions.ToArray());
            this.t++;
        }

        //カウントを取るために呼び変えしている
        protected abstract void Update(OptimizableFunction[] optimizableFunctions);
    }
}
