using NConstrictor;

namespace NChainer
{
    public class Convolution2D<T>
    {
        private PyObject _convolution2D;

        public Variable<T> W => _convolution2D["W"];
        public Variable<T> b => _convolution2D["b"];

        public Convolution2D(int inChannels, int outChannels, PyArray<T> kSize, PyArray<T> stride, PyArray<T> pad, bool nobias, PyArray<T> initialW, PyArray<T> initialBias)
        {
            _convolution2D = Chainer.Links["Convolution2D"].Call(inChannels, outChannels, kSize, stride, pad, nobias, initialW, initialBias);
            _convolution2D["cleargrads"].Call();            
        }

        public PyObject Forward(Variable<T> x)
        {
            return _convolution2D["forward"].Call(x);
        }
    }
}
