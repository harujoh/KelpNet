namespace KelpNet.Interface
{
    //コレから継承されたクラスが予測処理時に呼ばれる
    public interface IPredictableFunction
    {
        NdArray Predict(NdArray input);
    }
}
