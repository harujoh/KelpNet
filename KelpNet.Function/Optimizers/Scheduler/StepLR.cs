using System;

namespace KelpNet
{
    public class StepLR<T> : Scheduler<T> where T : unmanaged, IComparable<T>
    {
        public T Gamma;
        public int StepSize;

        StepLR(int stepSize, T gamma, int lastEpoch = 0) : base(lastEpoch)
        {
            Gamma = gamma;
            StepSize = stepSize;
        }

        public override T[] StepFunc(T[] inputParams)
        {
            if (Epoch == 0 || Epoch % StepSize != 0)
            {
                return inputParams;
            }

            T[] results = inputParams;

            switch (results)
            {
                case float[] resultsF:
                    for (int i = 0; i < results.Length; i++)
                    {
                        resultsF[i] *= (float)(object)Gamma;
                    }
                    break;

                case double[] resultsD:
                    for (int i = 0; i < results.Length; i++)
                    {
                        resultsD[i] *= (double)(object)Gamma;
                    }
                    break;
            }

            return results;
        }
    }
}
