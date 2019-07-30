using System;

namespace KelpNet
{
    public interface ISelectableSingleInputFunction : IFunction
    {
        Func<NdArray, NdArray> SingleInputForward { get; set; }
        Action<NdArray, NdArray> SingleOutputBackward { get; set; }
    }
}
