namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    public abstract class Function
    {
        //Backward用
        public NdArray PrevOutput;
        public NdArray PrevInput;

        public abstract NdArray Forward(NdArray x);
        public abstract NdArray Backward(NdArray gy);
    }
}
