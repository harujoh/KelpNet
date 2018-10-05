using System;
using System.Collections.Generic;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class AddBias : SingleInputFunction
    {
        const string FUNCTION_NAME = "AddBias";

        private int Axis;
        private NdArray Bias;

        public AddBias(int axis = 1, int[] biasShape = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
            this.Bias = new NdArray(biasShape);

            if (initialb != null)
            {
                Bias.Data = NdArray.GetArray(initialb);
            }

            this.Parameters = new[] { Bias };

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            int[] inputShape = x.Shape;
            int[] outputShape = this.Bias.Shape;

            List<int> shapeList = new List<int>();

            for (int i = 0; i < this.Axis; i++)
            {
                shapeList.Add(1);
            }

            shapeList.AddRange(outputShape);

            for (int i = 0; i < inputShape.Length - this.Axis - outputShape.Length; i++)
            {
                shapeList.Add(1);
            }

            int[] y1Shape = shapeList.ToArray();

            NdArray y1 = new Reshape(y1Shape).Forward(this.Bias)[0];
            NdArray y2 = new Broadcast(inputShape).Forward(y1)[0];

            return x + y2;
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
        }
    }
}
