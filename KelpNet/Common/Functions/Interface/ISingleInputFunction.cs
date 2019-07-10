using System;

namespace KelpNet
{
    public interface ISingleInputFunction : IFunction
    {
        Func<NdArray, NdArray> SingleInputForward { get; set; }
        Action<NdArray, NdArray> SingleOutputBackward { get; set; }
    }
}
