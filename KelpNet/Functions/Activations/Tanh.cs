using System;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : CompressibleActivation
    {
        const string FUNCTION_NAME = "Tanh";

        public Tanh(string name = FUNCTION_NAME, bool gpuEnable = false) : base(name, gpuEnable, FUNCTION_NAME)
        {
        }

        public override void ForwardActivate(ref Real x)
        {
            x = Math.Tanh(x);
        }

        public override void BackwardActivate(ref Real gy, Real y)
        {
            gy *= 1 - y * y;
        }
    }
}
