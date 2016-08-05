using System.Threading.Tasks;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    public abstract class Function
    {
        public abstract NdArray Forward(NdArray x);
        public abstract NdArray Backward(NdArray gy, NdArray PrevInput, NdArray PrevOutput);

        public virtual NdArray[] BatchForward(NdArray[] x)
        {
            NdArray[] y = new NdArray[x.Length];

            Parallel.For(0, x.Length, i =>
            {
                y[i] = Forward(x[i]);
            });

            return y;
        }

        public virtual NdArray[] BatchBackward(NdArray[] gy, NdArray[] PrevInput, NdArray[] PrevOutput)
        {
            NdArray[] gx = new NdArray[gy.Length];

            Parallel.For(0, gy.Length, i =>
            {
                gx[i] = Backward(gy[i], PrevInput[i], PrevOutput[i]);
            });

            return gx;
        }
    }
}
