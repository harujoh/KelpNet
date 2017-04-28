using KelpNet.Common;
using KelpNet.Common.Activations;

namespace KelpNet.Functions.Activations
{
    //アクティベーションが指定されない時にセットされるダミークラス
    class DummyActivation : Activation
    {
        public DummyActivation(string name = "Dummy", bool isGpu = false) : base(name, isGpu) { }

        public override void ForwardActivate(ref Real x) { }
        public override void BackwardActivate(ref Real gy, Real y) { }

        public override string ForwardActivateFunctionString { get; } = "void ForwardActivate(__global Real* gpuY){}";
        public override string BackwardActivateFunctionString { get; } = "void BackwardActivate(Real gpuY, Real* gpugX){}";
    }
}
