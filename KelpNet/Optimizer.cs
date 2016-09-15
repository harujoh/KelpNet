namespace KelpNet
{
    //Optimizerの基底クラス
    public abstract class Optimizer
    {
        //更新回数のカウント
        protected double UpdateCount = 1;
        private Function[] _functions;

        //更新回数のカウントを取りつつ更新処理を呼び出す
        public void Update()
        {
            this.DoUpdate(this._functions);
            this.UpdateCount++;
        }

        //カウントを取るために呼び変えしている
        protected abstract void DoUpdate(Function[] functions);

        //更新対象となる関数を保存
        public void SetFunctions(params Function[] functions)
        {
            this._functions = functions;
            this.Initialize(functions);
        }

        protected abstract void Initialize(Function[] functions);
    }
}
