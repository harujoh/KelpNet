using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Functions.Arrays;

namespace KelpNet.Functions.Connections
{
    public class Scale : SingleInputFunction
    {
        const string FUNCTION_NAME = "Scale";

        private int Axis;
        private NdArray W;
        private NdArray b;

        public Scale(int axis = 1, int[] wShape = null, bool biasTerm = false, int[] biasShape = null, string name = FUNCTION_NAME) : base(name)
        {
            Axis = axis;

            if (wShape != null)
            {
                W = new NdArray(wShape);
                W.Data = Enumerable.Repeat((Real)1.0, W.Data.Length).ToArray();

                if (biasTerm)
                {
                    b = new NdArray(wShape);
                }
            }
            else if (biasTerm)
            {
                b = new NdArray(biasShape);
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            int[] xShape = x.Shape;
            int[] yShape = W.Shape;

            List<int> shapeList = new List<int>();
            for (int i = 0; i < Axis; i++)
            {
                shapeList.Add(1);
            }
            shapeList.AddRange(yShape);
            for (int i = 0; i < xShape.Length - Axis - yShape.Length; i++)
            {
                shapeList.Add(1);
            }

            int[] y1Shape = shapeList.ToArray();
            NdArray y1 = W.Clone();
            y1 = new Reshape(y1Shape).Forward(y1)[0];

            NdArray b1 = b.Clone();
            b1 = new Reshape(y1Shape).Forward(b1)[0];

            NdArray y2 = new Broadcast(xShape).Forward(y1)[0];
            NdArray b2 = new Broadcast(xShape).Forward(b1)[0];

            return x * y2 + b2;
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
        }
    }
}
