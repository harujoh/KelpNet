using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class EmbedID : NeedPreviousDataFunction
    {
        public NdArray W;
        public NdArray gW;

        public EmbedID(int inputCount, int outputCount, Array initialW = null, string name = "EmbedID") : base(name)
        {
            this.W = NdArray.Zeros(inputCount, outputCount);
            this.gW = NdArray.ZerosLike(this.W);

            Parameters.Add(new OptimizeParameter(this.W, this.gW, Name + " W"));

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            OutputCount = outputCount;
            InputCount = inputCount;
        }

        protected override NdArray[] NeedPreviousForward(NdArray[] x)
        {
            NdArray[] resultArray = new NdArray[x.Length];

#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                NdArray result = NdArray.Zeros(x[i].Length, OutputCount);

                for (int j = 0; j < x[i].Length; j++)
                {
                    for (int k = 0; k < OutputCount; k++)
                    {
                        result.Data[j * OutputCount + k] = this.W.Data[(int)x[i].Data[j] * OutputCount + k];
                    }
                }

                resultArray[i] = result;
            }
#if !DEBUG
            );
#endif
            return resultArray;
        }

        protected override NdArray[] NeedPreviousBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput)
        {
#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                for (int j = 0; j < prevInput[i].Length; j++)
                {
                    for (int k = 0; k < OutputCount; k++)
                    {
                        this.gW.Data[(int)prevInput[i].Data[j] * OutputCount + k] += gy[i].Data[j + k];
                    }
                }
            }
#if !DEBUG
            );
#endif
            //これより上に層があるとエラーになる
            var dummy = new [] {NdArray.Zeros(1)};

            return dummy;
        }
    }
}
