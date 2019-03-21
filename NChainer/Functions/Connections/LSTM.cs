using NConstrictor;

namespace NChainer
{
    public class LSTM<T>
    {
        private PyObject _lstm;

        public Linear<T> upward => _lstm["upward"];
        public Linear<T> lateral => _lstm["lateral"];
        public Variable<T> c => _lstm["c"];
        public Variable<T> h => _lstm["h"];

        public LSTM(int inSize, int outSize, PyArray<T> lateralInit = default(PyArray<T>), PyArray<T> upwardInit = default(PyArray<T>), PyArray<T> biasInit = default(PyArray<T>), PyArray<T> forgetBiasInit = default(PyArray<T>))
        {
            _lstm = Chainer.Links["LSTM"].Call(inSize, outSize, lateralInit, upwardInit, biasInit, forgetBiasInit);
            _lstm["cleargrads"].Call();
        }

        public PyObject Forward(Variable<T> x)
        {
            return _lstm["forward"].Call(x);
        }
    }
}
