namespace KelpNet
{
    //バッチ処理専用の関数に使用（現在はBatchNorm専用）
    public interface IBatchFunction
    {
        NdArray[] BatchForward(NdArray[] x);
        NdArray[] BatchBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput);
    }
}
