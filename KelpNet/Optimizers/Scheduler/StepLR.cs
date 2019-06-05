namespace KelpNet
{
    public class StepLR : Scheduler
    {
        public Real Gamma;
        public int StepSize;

        StepLR(int stepSize, Real gamma, int lastEpoch = 0) : base(lastEpoch)
        {
            Gamma = gamma;
            StepSize = stepSize;
        }

        public override Real[] StepFunc(Real[] Params)
        {
            if (Epoch == 0 || Epoch % StepSize != 0)
                return Params;

            Real[] realts = Params;

            for (int i = 0; i < realts.Length; i++)
            {
                realts[i] *= Gamma;
            }

            return realts;
        }
    }
}
