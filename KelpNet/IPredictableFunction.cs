namespace KelpNet
{
    //予測処理用のインターフェース
    //コレを持つFunctionだけが予測処理時に呼ばれるようにする
    public interface IPredictableFunction
    {
        NdArray Predict(NdArray input);
    }
}
