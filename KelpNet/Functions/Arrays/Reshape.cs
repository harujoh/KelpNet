using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Arrays
{
    class Reshape : SingleInputFunction
    {
        const string FUNCTION_NAME = "Reshape";
        private int[] Shape;

        public Reshape(int[] shape, string name = FUNCTION_NAME) : base(name)
        {
            this.Shape = shape;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        NdArray ForwardCpu(NdArray val)
        {
            NdArray result = val.Clone();
            result.ParentFunc = this;
            result.Reshape(this.Shape);

            return result;
        }

        void BackwardCpu(NdArray y, NdArray x)
        {
            y.Grad = x.Grad.ToArray();
        }
    }
    
}
