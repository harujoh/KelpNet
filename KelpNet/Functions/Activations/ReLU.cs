using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : CompressibleActivation
    {
        const string FUNCTION_NAME = "ReLU";

        public ReLU(string name = FUNCTION_NAME, bool gpuEnable = false) : base(name, gpuEnable, FUNCTION_NAME)
        {
        }

        public override void ForwardActivate(ref Real x)
        {
            if (x < 0)
            {
                x = 0;
            }
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            if (y <= 0)
            {
                gy = 0;
            }
        }
    }
}
