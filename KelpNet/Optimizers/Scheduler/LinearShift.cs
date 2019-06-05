namespace KelpNet
{
    public class LinearShift : Scheduler
    {
        private Real[] ValueRange;
        private int[] TimeRange;
        private int t = 0;

        LinearShift(Real[] valueRange, int[] timeRange, int lastEpoch = 0) : base(lastEpoch)
        {
            ValueRange = valueRange;
            TimeRange = timeRange;
        }

        //LinearShiftでは元のパラメータを考慮しない
        public override Real[] StepFunc(Real[] Params)
        {
            t++;
            Real result;

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
                Real rate = (t - TimeRange[0]) / (TimeRange[1] - TimeRange[0]);
                result = ValueRange[0] + rate * (ValueRange[1] - ValueRange[0]);
            }

            Real[] realts = new Real[Params.Length];

            for (int i = 0; i < realts.Length; i++)
            {
                realts[i] = result;
            }

            return realts;
        }
    }
}
