using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class AddBias : SingleInputFunction
    {
        const string FUNCTION_NAME = "AddBias";

        private int Axis;
        private NdArray Bias;

        public AddBias(int axis = 0, int[] biasShape = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
            this.Bias = new NdArray(biasShape);

            if (initialb != null)
            {
                Bias.Data = Real.ToRealArray(initialb);
            }

            this.Parameters = new[] { Bias };
        }

        public override NdArray SingleInputForward(NdArray x)
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

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            //AddBiasとして必要な処理はない
        }
    }
}
