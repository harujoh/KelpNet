namespace KelpNet
{
    public interface IParallelizable
    {
        void CreateKernel();
        bool SetGpuEnable(bool enable);
    }
}
