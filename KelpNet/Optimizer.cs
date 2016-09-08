using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの基底クラス
    public abstract class Optimizer
    {
        //更新回数のカウント
        protected double t = 1;

        //更新回数のカウントを取りつつ更新処理を呼び出す
        public void Update(List<Function> functions)
        {
            this.DoUpdate(functions);
            this.t++;
        }

        //カウントを取るために呼び変えしている
        protected abstract void DoUpdate(List<Function> functions);

        //ネットワークの大きさをOptimizerに保存
        public abstract void Initialize(FunctionStack fs);
    }
}
