using System;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    [Serializable]
    public class GradientClipping<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Threshold;

        public GradientClipping(T threshold)
        {
            this.Threshold = threshold;
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new GradientClippingParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    public class GradientClippingParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly GradientClipping<T> optimizer;

        public GradientClippingParameter(NdArray<T> functionParameter, GradientClipping<T> optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;

            switch (this)
            {
                case GradientClippingParameter<float> momentumSgdParameterF:
                    momentumSgdParameterF.UpdateFunctionParameters = () => GradientClippingParameterF.UpdateFunctionParameters(momentumSgdParameterF.optimizer.Threshold, momentumSgdParameterF.FunctionParameter);
                    break;

                case GradientClippingParameter<double> momentumSgdParameterD:
                    momentumSgdParameterD.UpdateFunctionParameters = () => GradientClippingParameterD.UpdateFunctionParameters(momentumSgdParameterD.optimizer.Threshold, momentumSgdParameterD.FunctionParameter);
                    break;
            }
        }

    }
#endif

#if DOUBLE
    public static class GradientClippingParameterD
#else
    public static class GradientClippingParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real Threshold, NdArray<Real> FunctionParameter)
        {
            //_sum_sqnorm
            Real s = 0;

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                s += FunctionParameter.Grad[i] * FunctionParameter.Grad[i];
            }

            Real norm = KelpMath.Sqrt(s);
            Real rate = Threshold / norm;

            if (rate < 1)
            {
                for (int i = 0; i < FunctionParameter.Data.Length; i++)
                {
                    FunctionParameter.Grad[i] *= rate;
                }
            }
        }
    }
}
