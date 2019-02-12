using System;
using KelpNet;
using NConstrictor;

namespace NChainer
{
    struct Linear
    {
        private PyObject _linear;

        public Linear(int inSize, int outSize, bool noBias, Array initialW, Array initialBias)
        {
            _linear = Chainer.Links["Linear"].Call(inSize, outSize, noBias, (PyArray<Real>)Real.ToBaseArray(initialW), (PyArray<Real>)Real.ToBaseArray(initialBias));
        }

        public PyObject Forward<T>(Variable<T> x, int nBatchAxes = 1)
        {
            return Python.GetNamelessObject(_linear["forward"].Call(x, nBatchAxes));
        }
    }
}
