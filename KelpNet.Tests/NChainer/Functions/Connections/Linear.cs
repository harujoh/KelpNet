using NConstrictor;

namespace NChainer
{
    struct Linear<T>
    {
        private PyObject _linear;

        public Variable<T> W => _linear["W"];
        public Variable<T> b => _linear["b"];

        public Linear(int inSize, int outSize, bool noBias, PyArray<T> initialW, PyArray<T> initialBias)
        {
            _linear = Chainer.Links["Linear"].Call(inSize, outSize, noBias, initialW, initialBias);
            _linear["cleargrads"].Call();
        }

        public PyObject Forward(Variable<T> x, int nBatchAxes = 1)
        {
            return Python.GetNamelessObject(_linear["forward"].Call(x, nBatchAxes));
        }
    }
}
