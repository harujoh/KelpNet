namespace KelpNet
{
    //Optimizerの基底クラス
    public interface IOptimizer
    {
        //更新処理
        void Update(OptimizeParameter parameter);
        IOptimizer Initialise(OptimizeParameter parameter);
    }
}
