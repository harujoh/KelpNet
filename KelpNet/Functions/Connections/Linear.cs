using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousDataFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name)
        {
            this.W = NdArray.Zeros(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);
            Parameters.Add(new OptimizeParameter(this.W, this.gW, Name + " W"));

            //Zeroバイアス
            this.b = NdArray.Zeros(outputCount);

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            if (!noBias)
            {
                this.gb = NdArray.ZerosLike(this.b);

                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                Parameters.Add(new OptimizeParameter(this.b, this.gb, Name + " b"));
            }

            OutputCount = outputCount;
            InputCount = inputCount;
        }

        protected override NdArray[] NeedPreviousForward(NdArray[] x)
        {
            NdArray[] y = new NdArray[x.Length];

#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                double[] output = new double[OutputCount];

                for (int j = 0; j < OutputCount; j++)
                {
                    for (int k = 0; k < InputCount; k++)
                    {
                        output[j] += x[i].Data[k] * this.W.Get(j, k);
                    }

                    output[j] += this.b.Data[j];
                }

                y[i] = NdArray.FromArray(output);
            }
#if !DEBUG
            );
#endif

            return y;
        }

        protected override NdArray[] NeedPreviousBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput)
        {
            NdArray[] gx = new NdArray[gy.Length];

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                for (int j = 0; j < prevInput[i].Length; j++)
                {
                    for (int k = 0; k < gy[i].Length; k++)
                    {
                        this.gW.Data[this.gW.GetIndex(k, j)] += prevInput[i].Data[j] * gy[i].Data[k];
                    }
                }

                double[] gxData = new double[InputCount];

                for (int j = 0; j < this.W.Shape[0]; j++)
                {
                    for (int k = 0; k < this.W.Shape[1]; k++)
                    {
                        gxData[k] += this.W.Get(j, k) * gy[i].Data[j];
                    }
                }

                if (this.gb != null)
                {
                    for (int j = 0; j < gy[i].Length; j++)
                    {
                        this.gb.Data[j] += gy[i].Data[j];
                    }
                }

                gx[i] = new NdArray(gxData, new[] { 1, InputCount });
            }
#if !DEBUG
            );
#endif

            return gx;
        }
    }
}
