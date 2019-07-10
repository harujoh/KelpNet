using System.Collections.Generic;

namespace KelpNet
{
    public interface IFunction
    {
        List<NdArray[]> PrevInputs { get; set; }

        void Backward(params NdArray[] ys);

    }
}
