using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class SoftmaxCrossEntropy : ILossFunction
    {
        public BatchArray Evaluate(BatchArray input, BatchArray teachSignal, out Real loss)
        {
            //Real[] localloss = new Real[input.BatchCount];
            Real localloss = 0;
            Real[] gx = new Real[input.Data.Length];

            for (int b = 0; b < input.BatchCount; b++)
            {
                Real maxIndex = 0;

                for (int i = 0; i < teachSignal.Length; i++)
                {
                    if (maxIndex < teachSignal.Data[i + b * teachSignal.Length])
                    {
                        maxIndex = teachSignal.Data[i + b * teachSignal.Length];
                    }
                }

                Real[] logY = new Real[input.Length];
                Real y = 0;
                Real m = input.Data[b * input.Length];

                for (int i = 1; i < input.Length; i++)
                {
                    if (m < input.Data[i + b * input.Length])
                    {
                        m = input.Data[i + b * input.Length];
                    }
                }

                for (int i = 0; i < input.Length; i++)
                {
                    y += (Real)Math.Exp(input.Data[i + b * input.Length] - m);                    
                }

                m += (Real)Math.Log(y);

                for (int i = 0; i < input.Length; i++)
                {
                    logY[i] = input.Data[i + b * input.Length] - m;
                }

                localloss += -logY[(int)maxIndex];


                for (int i = 0; i < logY.Length; i++)
                {
                    gx[i + b * input.Length] = (Real)Math.Exp(logY[i]);
                }

                gx[(int)maxIndex + b * input.Length] -= 1;
            }

            loss = localloss / input.BatchCount;

            return BatchArray.Convert(gx, input.Shape, input.BatchCount);
        }
    }
}
