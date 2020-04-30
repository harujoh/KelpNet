using System;

namespace KelpNet
{
    public class LinearShift<T> : Scheduler<T> where T : unmanaged, IComparable<T>
    {
        private T[] ValueRange;
        private int[] TimeRange;
        private int t = 0;

        LinearShift(T[] valueRange, int[] timeRange, int lastEpoch = 0) : base(lastEpoch)
        {
            ValueRange = valueRange;
            TimeRange = timeRange;
        }

        //LinearShiftでは元のパラメータを考慮しない
        public override T[] StepFunc(T[] inputParams)
        {
            t++;
            T result = default;

            if (t <= TimeRange[0])
            {
                result = ValueRange[0];
            }
            else if (t >= TimeRange[1])
            {
                result = ValueRange[1];
            }
            else
            {
                switch (ValueRange)
                {
                    case float[] valueRangeF:
                        float rateF = (float)(t - TimeRange[0]) / (TimeRange[1] - TimeRange[0]);
                        result = (T)(object)(valueRangeF[0] + rateF * (valueRangeF[1] - valueRangeF[0]));
                        break;
                    case double[] valueRangeF:
                        double rateD = (double)(t - TimeRange[0]) / (TimeRange[1] - TimeRange[0]);
                        result = (T)(object)(valueRangeF[0] + rateD * (valueRangeF[1] - valueRangeF[0]));
                        break;
                }
            }

            T[] realts = new T[inputParams.Length];

            for (int i = 0; i < realts.Length; i++)
            {
                realts[i] = result;
            }

            return realts;
        }
    }
}
