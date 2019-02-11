using NConstrictor;

namespace NChainer
{
    struct Linear
    {
        private PyObject _linear;

        public Linear(int inSize, int outSize, bool noBias, PyObject initialW, PyObject initialBias)
        {
            _linear = Chainer.Links["Linear"].Call(inSize, outSize, noBias, initialW, initialBias);
        }

        public PyObject Forward(Variable x, int nBatchAxes = 1)
        {
            return Python.GetNamelessObject(_linear["forward"].Call(x, nBatchAxes));
        }
    }
}
