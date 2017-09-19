namespace KelpNet.Common.Functions
{
    public abstract class CompressibleFunction : NeedPreviousInputFunction
    {
        protected CompressibleActivation Activation;

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
