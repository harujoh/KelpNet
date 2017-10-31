using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Mathmetrics.BasicMath
{
    public class Div : DualInputFunction
    {
        private const string FUNCTION_NAME = "Div";

        public Div(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] / b.Data[i];
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = y.Grad[i] / b.Data[i];
                a.Grad[i] += gx;
                b.Grad[i] += -gx * a.Data[i] / b.Data[i];
            }
        }
    }

    //右辺が定数
    public class DivConst : DualInputFunction
    {
        private const string FUNCTION_NAME = "DivConst";

        public DivConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] / val;
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            Real val = b.Data[0];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i] / val;
            }
        }
    }

    //左辺が定数
    public class ConstDiv : DualInputFunction
    {
        private const string FUNCTION_NAME = "ConstDiv";

        public ConstDiv(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = a.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = val / b.Data[i];
            }

            return new NdArray(resultData, b.Shape, b.BatchCount, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            Real val = a.Data[0];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = y.Grad[i] / b.Data[i];
                b.Grad[i] += -gx * val / b.Data[i];
            }
        }
    }

}
