using NConstrictor;

namespace NChainer
{
    public struct Linear<T>
    {
        private PyObject _linear;

        public Variable<T> W => _linear["W"];
        public Variable<T> b => _linear["b"];

        public Linear(int inSize, int outSize, bool noBias = false, PyArray<T> initialW = default(PyArray<T>), PyArray<T> initialBias = default(PyArray<T>))
        {
            _linear = Chainer.Links["Linear"].Call(inSize, outSize, noBias, initialW, initialBias);
            _linear["cleargrads"].Call();
        }

        public PyObject Forward(Variable<T> x, int nBatchAxes = 1)
        {
            return _linear["forward"].Call(x, nBatchAxes);
        }

        public static implicit operator PyObject(Linear<T> linear)
        {
            return linear._linear;
        }
    }
}
