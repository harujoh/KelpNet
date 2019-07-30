namespace KelpNet
{
    public class Add : DualInputFunction
    {
        private const string FUNCTION_NAME = "Add";

        public Add(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] + b.Data[i];
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

    public class AddConst : DualInputFunction
    {
        private const string FUNCTION_NAME = "AddConst";

        public AddConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] + val;
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

    //ConstAddは呼び出し元でひっくり返しているため不要
}