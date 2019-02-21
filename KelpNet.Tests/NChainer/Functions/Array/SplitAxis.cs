using NConstrictor;

namespace NChainer
{
    class SplitAxis<T>
    {
        private PyObject _splitAxis;

        public SplitAxis()
        {
            _splitAxis = Chainer.Functions["split_axis"];
        }

        public PyObject[] Forward(PyObject x, PyArray<int> indices, int axis = 1)
        {
            return PyTuple.UnPack(_splitAxis.Call(x, indices, axis));
        }
    }
}
