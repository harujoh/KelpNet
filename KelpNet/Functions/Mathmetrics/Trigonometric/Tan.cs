using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet.Mathmetricsetrics
#else
using Real = System.Single;
namespace KelpNet.Mathmetrics
#endif
{
    public class Tan : SingleInputFunction
    {
        private const string FUNCTION_NAME = "Tan";

        public Tan(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = (Real)Math.Tan(x.Data[i]);
            }

            return new NdArray(resultData, x.Shape, x.BatchCount, this);
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = (Real)Math.Cos(x.Data[i]);
                x.Grad[i] += 1 / (gx * gx) * y.Grad[i];
            }
        }
    }
}
