using NConstrictor;

namespace NChainer
{
    class Concat<T>
    {
        private PyObject _concat;

        private int axis;

        public Concat(int axis = 1)
        {
            this.axis = axis;
            _concat = Chainer.Functions["concat"];
        }

        public PyObject Forward(PyObject x)
        {
            return _concat.Call(x, axis);
        }
    }
}
