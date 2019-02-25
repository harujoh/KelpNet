using NConstrictor;

namespace NChainer
{
    public class SoftmaxCrossEntropy<T>
    {
        private PyObject _softmaxCrossEntropy;

        public SoftmaxCrossEntropy()
        {
            _softmaxCrossEntropy = Chainer.Functions["softmax_cross_entropy"];
        }

        public PyObject Forward(PyObject x0, PyObject x1)
        {
            return _softmaxCrossEntropy.Call(x0, x1);
        }
    }
}
