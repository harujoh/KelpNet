using System;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleFunction : NeedPreviousInputFunction
    {
        public CompressibleActivation Activation { get; protected set; }

        protected CompressibleFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
        }

        public void SetActivation(CompressibleActivation activation)
        {
            this.Activation = activation;

            if (this.IsGpu)
            {
                CreateKernel();
            }
        }
    }
}
