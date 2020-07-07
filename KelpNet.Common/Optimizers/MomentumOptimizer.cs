using System;
using System.Collections.Generic;

namespace KelpNet
{
    public class MomentumOptimizer<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public List<T[]> var = new List<T[]>();
    }
}
