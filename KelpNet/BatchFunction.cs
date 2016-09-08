namespace KelpNet
{
    public interface IBatchFunction
    {
        NdArray[] BatchForward(NdArray[] x);
        NdArray[] BatchBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput);
    }
}
