using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet
{
    [Serializable]
    public class MultiplyScale<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "MultiplyScale";

        private int Axis;
        public NdArray<T> Weight;
        public NdArray<T> Bias;
        public bool BiasTerm = false;

        public MultiplyScale(int axis = 1, int[] wShape = null, bool biasTerm = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
            this.BiasTerm = biasTerm;

#if DEBUG
            if (wShape == null)
            {
                if (biasTerm)
                {
                    throw new Exception("Biasのみでの使用はAddBiasを使用してください");
                }
                else
                {
                    throw new Exception("パラメータの設定が正しくありません");
                }
            }
#endif
            this.Weight = new NdArray<T>(wShape);
            this.Parameters = new NdArray<T>[biasTerm ? 2 : 1];

            if (initialW == null)
            {
                this.Weight.Data = Enumerable.Repeat((Real<T>)1.0, Weight.Data.Length).ToArray();
            }
            else
            {
                this.Weight.Data = Real<T>.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (biasTerm)
            {
                this.Bias = new NdArray<T>(wShape);

                if (initialb != null)
                {
                    this.Bias.Data = Real<T>.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> x)
        {
            int[] inputShape = x.Shape;
            int[] outputShape = this.Weight.Shape;

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

            int[] preShape = shapeList.ToArray();

            NdArray<T> y1 = new Reshape<T>(preShape).Forward(this.Weight)[0];
            NdArray<T> y2 = new Broadcast<T>(inputShape).Forward(y1)[0];

            if (BiasTerm)
            {
                NdArray<T> b1 = new Reshape<T>(preShape).Forward(this.Bias)[0];
                NdArray<T> b2 = new Broadcast<T>(inputShape).Forward(b1)[0];

                return x * y2 + b2;
            }
            else
            {
                return x * y2;
            }
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            //MultiplyScaleとしては処理はない
        }
    }
}
