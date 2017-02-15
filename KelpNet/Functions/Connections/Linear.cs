using System;
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

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name, inputCount, outputCount)
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

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] output = new double[OutputCount * x.BatchCount];

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    output[i + batchCount * this.OutputCount] += this.b.Data[i];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        output[i + batchCount * this.OutputCount] += x.Data[j + batchCount * x.Length] * this.W.Data[i * this.InputCount + j];
                    }
                }
            }

            return BatchArray.Convert(output, new[] { OutputCount }, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            double[] gxData = new double[prevInput.Data.Length];

            for (int b = 0; b < gy.BatchCount; b++)
            {
                int indexOffset = 0;

                for (int i = 0; i < gy.Length; i++)
                {
                    double gyData = gy.Data[i + b * gy.Length];
                    this.gb.Data[i] += gyData;

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.gW.Data[indexOffset] += prevInput.Data[j + b * prevInput.Length] * gyData;
                        gxData[j + b * prevInput.Length] += this.W.Data[indexOffset++] * gyData;
                    }
                }
            }

            return BatchArray.Convert(gxData, prevInput.Shape, prevInput.BatchCount);
        }
    }
}
