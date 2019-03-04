using NConstrictor;

namespace NChainer
{
    public class Deconvolution2D<T>
    {
        private PyObject _deconvolution2D;

        public Variable<T> W => _deconvolution2D["W"];
        public Variable<T> b => _deconvolution2D["b"];

        public Deconvolution2D(int inChannels, int outChannels, PyArray<T> kSize, PyArray<T> stride, PyArray<T> pad, bool nobias, PyObject[] outSize, PyArray<T> initialW, PyArray<T> initialBias)
        {
            _deconvolution2D = Chainer.Links["Deconvolution2D"].Call(inChannels, outChannels, kSize, stride, pad, nobias, PyTuple.Pack(outSize), initialW, initialBias);
            _deconvolution2D["cleargrads"].Call();
        }

        public PyObject Forward(Variable<T> x)
        {
            return _deconvolution2D["forward"].Call(x);
        }
    }
}
