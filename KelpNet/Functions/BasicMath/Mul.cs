using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.BasicMath
{
    public class Mul : DualInputFunction
    {
        private const string FUNCTION_NAME = "Mul";

        public Mul(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] * b.Data[i];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += b.Data[i] * y.Grad[i];
                b.Grad[i] += a.Data[i] * y.Grad[i];
            }
        }
    }

    public class MulConst : DualInputFunction
    {
        private const string FUNCTION_NAME = "MulConst";

        public MulConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] * b.Data[0];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += b.Data[0] * y.Grad[i];
            }
        }
    }

}
