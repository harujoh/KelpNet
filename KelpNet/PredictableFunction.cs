namespace KelpNet
{
    //予測処理用のインターフェース
    //コレを持つFunctionだけが予測処理時に呼ばれるようにする
    public abstract class PredictableFunction : Function
    {
        public virtual NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
