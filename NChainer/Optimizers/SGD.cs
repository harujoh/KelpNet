using NConstrictor;

namespace NChainer
{
    public class SGD<T>
    {
        private PyObject _sgd;

        public SGD(double learningRate = 0.01)
        {
            _sgd = Chainer.Optimizers["SGD"].Call(learningRate);
        }

        public void Setup(PyObject link)
        {
            _sgd["setup"].Call(link);
        }

        public void Update()
        {
            _sgd["update"].Call();
        }
    }
}
