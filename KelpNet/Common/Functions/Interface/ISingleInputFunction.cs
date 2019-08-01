namespace KelpNet
{
    public interface ISingleInputFunction : IFunction
    {
        NdArray SingleInputForward(NdArray x);
        void SingleOutputBackward(NdArray y, NdArray x);
    }
}
