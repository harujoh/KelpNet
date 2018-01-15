namespace KelpNet.Common.Functions
{
    public interface IParallelizable
    {
        void CreateKernel();
        bool SetGpuEnable(bool enable);
    }
}
