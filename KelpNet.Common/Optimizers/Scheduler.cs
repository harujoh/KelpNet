using System;

namespace KelpNet
{
    public abstract class Scheduler<T> where T : unmanaged, IComparable<T>
    {
        public int Epoch = 0;

        protected Scheduler(int lastEpoch = 0)
        {
            Epoch = lastEpoch;
        }

        public T Step(T Param)
        {
            return Step(new[] { Param })[0];
        }

        public T[] Step(T[] Params)
        {
            Epoch++;
            return StepFunc(Params);
        }

        public abstract T[] StepFunc(T[] Params);
    }
}
