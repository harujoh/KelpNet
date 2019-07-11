using System.Collections.Generic;

namespace KelpNet
{
    public interface IFunction
    {
        string Name { get; set; }

        NdArray[] Parameters { get; set; }

        Optimizer[] Optimizers { get; set; }

        List<NdArray[]> PrevInputs { get; set; }
        List<NdArray[]> UsedPrevInputs { get; set; }

        string[] InputNames { get; set; }
        string[] OutputNames { get; set; }

        NdArray[] Forward(params NdArray[] xs);
        NdArray[] Predict(params NdArray[] xs);
        void Backward(params NdArray[] ys);
    }
}
