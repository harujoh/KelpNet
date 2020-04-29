namespace KelpNet.CL
{
    public interface IParallelizable
    {
        string FunctionName { get; }
        string KernelSource { get; }

        bool IsParallel { get; set; }

        bool SetParallel(bool enable);
    }
}
