using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.BasicMath
{
    public class Add : DualInputFunction
    {
        private const string FUNCTION_NAME = "Add";

        public Add(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] + b.Data[i];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }
}
