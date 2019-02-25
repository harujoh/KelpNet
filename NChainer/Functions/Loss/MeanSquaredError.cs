using NConstrictor;

namespace NChainer
{
    public class MeanSquaredError<T>
    {
        private PyObject _meanSquaredError;

        public MeanSquaredError()
        {
            _meanSquaredError = Chainer.Functions["mean_squared_error"];
        }

        public PyObject Forward(PyObject x0, PyObject x1)
        {
            return _meanSquaredError.Call(x0, x1);
        }
    }
}
