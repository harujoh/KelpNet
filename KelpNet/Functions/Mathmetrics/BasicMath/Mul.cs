namespace KelpNet
{
    public class Mul : DualInputFunction
    {
        private const string FUNCTION_NAME = "Mul";

        public Mul(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] * b.Data[i];
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
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
        }

        protected override NdArray DualInputForward(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] * val;
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        protected override void DualOutputBackward(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += b.Data[0] * y.Grad[i];
            }
        }
    }

}
