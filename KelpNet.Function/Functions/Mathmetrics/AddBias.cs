using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

namespace KelpNet
{
    [Serializable]
    public class AddBias<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "AddBias";

        private int Axis;
        private NdArray<T> Bias;

        public AddBias(int axis = 0, int[] biasShape = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
            this.Bias = new NdArray<T>(biasShape);

            if (initialb != null)
            {
                Buffer.BlockCopy(initialb, 0, this.Bias.Data, 0, initialb.Length * Marshal.SizeOf<T>());
            }

            this.Parameters = new[] { Bias };

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.SingleInputForward = this.AddBiasForward;
            base.SingleOutputBackward = (y, x) => { };//AddBiasとして必要な処理はない
        }

        public NdArray<T> AddBiasForward(NdArray<T> x)
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

            NdArray<T> y1 = new Reshape<T>(y1Shape).Forward(this.Bias)[0];
            NdArray<T> y2 = new Broadcast<T>(inputShape).Forward(y1)[0];

            return x + y2;
        }
    }
}
