using NConstrictor;

namespace NChainer
{
    public class EmbedID<T>
    {
        private PyObject _embedID;

        public Variable<T> W => _embedID["W"];

        public EmbedID(int inSize, int outSize, PyArray<T> initialW = default(PyArray<T>))
        {
            _embedID = Chainer.Links["EmbedID"].Call(inSize, outSize, initialW);
            _embedID["cleargrads"].Call();
        }

        public PyObject Forward(Variable<int> x)
        {
            return _embedID["forward"].Call(x);
        }
    }
}
