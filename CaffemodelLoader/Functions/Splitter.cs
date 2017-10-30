using KelpNet.Common;
using KelpNet.Common.Functions;

namespace CaffemodelLoader
{
    public class Splitter : MultiOutputFunction
    {
        const string FUNCTION_NAME = "Splitter";

        public Splitter(string name = FUNCTION_NAME) : base(name)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray[] ForwardCpu(NdArray x)
        {
            return new[] {x, x};
        }

        private void BackwardCpu(NdArray[] ys, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += ys[0].Grad[i] + ys[1].Grad[i];
            }
        }
    }
}
