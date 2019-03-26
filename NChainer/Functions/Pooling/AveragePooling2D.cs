using NConstrictor;

namespace NChainer
{
    public class AveragePooling2D<T>
    {
        private PyObject _maxPooling2D;

        private PyArray<T> _kSize;
        private PyArray<T> _stride;
        private PyArray<T> _pad;

        public AveragePooling2D(PyArray<T> kSize, PyArray<T> stride, PyArray<T> pad)
        {
            this._kSize = kSize;
            this._stride = stride;
            this._pad = pad;
            _maxPooling2D = Chainer.Functions["average_pooling_2d"];
        }

        public PyObject Forward(Variable<T> x)
        {
            return _maxPooling2D.Call(x, _kSize, _stride, _pad);
        }

    }
}
