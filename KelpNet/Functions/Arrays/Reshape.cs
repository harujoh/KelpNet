using System.Linq;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    class Reshape : SingleInputFunction
    {
        const string FUNCTION_NAME = "Reshape";
        public int[] Shape;

        public Reshape(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
