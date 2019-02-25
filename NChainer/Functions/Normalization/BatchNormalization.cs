using System;
using NConstrictor;

namespace NChainer
{
    public struct BatchNormalization<T>
    {
        private PyObject _batchNormalization;

        public Variable<T> gamma
        {
            get { return _batchNormalization["gamma"]; }
            set { _batchNormalization["gamma"] = value; }
        }

        public Variable<T> beta
        {
            get { return _batchNormalization["beta"]; }
            set { _batchNormalization["beta"] = value; }
        }

        public PyArray<T> avgMean
        {
            get { return _batchNormalization["avg_mean"]; }
            set { _batchNormalization["avg_mean"] = value; }
        }

        public PyArray<T> avgVar
        {
            get { return _batchNormalization["avg_var"]; }
            set { _batchNormalization["avg_var"] = value; }
        }

        public PyObject N => _batchNormalization["N"];
        public PyObject decay => _batchNormalization["decay"];
        public PyObject eps => _batchNormalization["eps"];

        public BatchNormalization(int size, float decay = 0.9f, float eps = 2e-05f, Type dtype = null, bool useGamma = true, bool useBeta = true, int initialGamma = 1, int initialBeta = 0, int? axis = null, int initialAvgMean = 0, int initialAvgVar = 1)
        {
            _batchNormalization = Chainer.Links["BatchNormalization"].Call(
                size,
                decay,
                eps,
                Dtype.GetDtype(dtype == null ? typeof(T) : dtype),
                useGamma,
                useBeta,
                initialGamma,
                initialBeta,
                axis == null ? Py.None : (PyObject)axis,
                initialAvgMean,
                initialAvgVar
            );
            _batchNormalization["cleargrads"].Call();
        }

        public PyObject Forward(Variable<T> x, bool fineTune = false)
        {
            return _batchNormalization["forward"].Call(new PyObject[] { x }, new PyDict("finetune", fineTune));
        }
    }
}
