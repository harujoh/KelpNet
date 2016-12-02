using System;

namespace KelpNet.Optimizers
{
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping : IOptimizer
    {
        public double Threshold;

        public GradientClipping(double threshold)
        {
            this.Threshold = threshold;
        }

        //パラメーターがないのでクローンの必要がない
        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return this;
        }

        public void Update(OptimizeParameter parameter)
        {
            //_sum_sqnorm
            double s = 0.0;

            for (int i = 0; i < parameter.Length; i++)
            {
                for (int j = 0; j < parameter.Length; j++)
                {
                    s += Math.Pow(parameter.Grad.Data[j], 2);
                }
            }

            double norm = Math.Sqrt(s);
            double rate = this.Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < parameter.Length; i++)
                {
                    parameter.Grad.Data[i] *= rate;
                }
            }
        }
    }
}
