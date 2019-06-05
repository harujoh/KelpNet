namespace KelpNet
{
    public abstract class Scheduler
    {
        public int Epoch = 0;

        protected Scheduler(int lastEpoch = 0)
        {
            Epoch = lastEpoch;
        }

        public Real Step(Real Param)
        {
            return Step(new[] { Param })[0];
        }

        public Real[] Step(Real[] Params)
        {
            Epoch++;
            return StepFunc(Params);
        }

        public abstract Real[] StepFunc(Real[] Params);
    }
}
