namespace KelpNet
{
    public interface IParallelizable
    {
        bool IsParallel { get; set; }

        void InitParallel();
        bool SetParallel(bool enable);
    }
}
