using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

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

        public MultiplyScale(int axis = 0, int[] wShape = null, bool biasTerm = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
                this.Weight.Fill((TVal<T>)1.0);
            }
            else
            {
                Buffer.BlockCopy(initialW, 0, this.Weight.Data, 0, initialW.Length * Marshal.SizeOf<T>());
            }

            this.Parameters[0] = this.Weight;

            if (biasTerm)
            {
                this.Bias = new NdArray<T>(wShape);

                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.Bias.Data, 0, initialb.Length * Marshal.SizeOf<T>());
                }

                this.Parameters[1] = this.Bias;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.SingleInputForward = this.MultiplyScaleForward;
            base.SingleOutputBackward = (y, x) => { };//MultiplyScaleとして必要な処理はない
        }

        public NdArray<T> MultiplyScaleForward(NdArray<T> x)
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
    }
}
