using NConstrictor;

namespace NChainer
{
    public class MaxPooling2D<T>
    {
        private PyObject _maxPooling2D;

        private PyArray<T> _kSize;
        private PyArray<T> _stride;
        private PyArray<T> _pad;
        private bool _coverAll;

        public MaxPooling2D(PyArray<T> kSize, PyArray<T> stride, PyArray<T> pad, bool coverAll = true)
        {
            this._kSize = kSize;
            this._stride = stride;
            this._pad = pad;
            this._coverAll = coverAll;
            _maxPooling2D = Chainer.Functions["max_pooling_2d"];
        }

        public PyObject Forward(Variable<T> x)
        {
            return _maxPooling2D.Call(x, _kSize, _stride, _pad, _coverAll);
        }

    }
}
