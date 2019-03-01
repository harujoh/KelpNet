using NConstrictor;

namespace NChainer
{
    public class Swish<T>
    {
        private PyObject _swish;
        public Variable<T> beta => _swish["beta"];

        public Swish(int[] betaShape, double beta = 1.0)
        {
            PyObject[] beta_shape = new PyObject[betaShape.Length];
            for (int i = 0; i < betaShape.Length; i++)
            {
                beta_shape[i] = betaShape[i];
            }

            _swish = Chainer.Links["Swish"].Call(PyTuple.Pack(beta_shape), beta);
            _swish["cleargrads"].Call();
        }

        public PyObject Forward(PyObject x)
        {
            return _swish.Call(x);
        }
    }
}
