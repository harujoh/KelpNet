using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear", bool isParallel = true) : base(name, inputCount, outputCount,isParallel)
        {
            this.W = NdArray.Zeros(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            //最初からバイアスをコピーして入れておく
            double[] output = this.b.Data.ToArray();
            int indexOffset = 0;

            for (int i = 0; i < this.OutputCount; i++)
            {
                for (int j = 0; j < this.InputCount; j++)
                {
                    output[i] += x.Data[j] * this.W.Data[indexOffset++];
                }
            }

            return NdArray.Convert(output);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput)
        {
            double[] gxData = new double[this.InputCount];
            int indexOffset = 0;

            for (int i = 0; i < gy.Length; i++)
            {
                double gyData = gy.Data[i];
                this.gb.Data[i] += gyData;

                for (int j = 0; j < this.InputCount; j++)
                {
                    this.gW.Data[indexOffset] += prevInput.Data[j] * gyData;
                    gxData[j] += this.W.Data[indexOffset++] * gyData;
                }
            }

            return NdArray.Convert(gxData, prevInput.Shape);
        }
    }
}
