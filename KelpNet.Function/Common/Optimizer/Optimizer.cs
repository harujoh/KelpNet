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
            if (optimizer.OptimizerParameters[0].FunctionParameter.TrainCount > 0)
            {
                optimizer.UpdateCount++;

                for (int i = 0; i < optimizer.OptimizerParameters.Count; i++)
                {
                    //傾きの割引を実行
                    optimizer.OptimizerParameters[i].FunctionParameter.Reduce();

                    optimizer.OptimizerParameters[i].UpdateFunctionParameters();

                    optimizer.OptimizerParameters[i].FunctionParameter.InitGrad();

                    //カウンタをリセット
                    optimizer.OptimizerParameters[i].FunctionParameter.TrainCount = 0;
                }
            }

            optimizer.ResetState();
        }
    }
}
