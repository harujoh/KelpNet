namespace KelpNet
{
    public class Sub : DualInputFunction
    {
        private const string FUNCTION_NAME = "Sub";

        public Sub(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - b.Data[i];
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

    //右辺が定数
    public class SubConst : DualInputFunction
    {
        private const string FUNCTION_NAME = "SubConst";

        public SubConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - val;
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
            }
        }
    }

    //左辺が定数
    public class ConstSub : DualInputFunction
    {
        private const string FUNCTION_NAME = "ConstSub";

        public ConstSub(string name = FUNCTION_NAME) : base(name)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = a.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = val - b.Data[i];
            }

            return new NdArray(resultData, b.Shape, b.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

}
