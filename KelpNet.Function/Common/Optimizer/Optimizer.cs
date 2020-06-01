#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if DOUBLE
    public static class OptimizerD
#else
    public static class OptimizerF
#endif
    {
        public static void Update(Optimizer<Real> optimizer)
        {
            if (optimizer.FunctionParameters[0].TrainCount > 0)
            {
                optimizer.UpdateCount++;

                for (int i = 0; i < optimizer.FunctionParameters.Count; i++)
                {
                    //傾きの割引を実行
                    optimizer.FunctionParameters[i].Reduce();

                    optimizer.UpdateFunctionParameters(i);

                    optimizer.FunctionParameters[i].InitGrad();

                    //カウンタをリセット
                    optimizer.FunctionParameters[i].TrainCount = 0;
                }
            }

            optimizer.ResetState();
        }
    }
}
