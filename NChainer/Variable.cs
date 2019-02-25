using System.Runtime.CompilerServices;
using NConstrictor;

namespace NChainer
{
    public struct Variable<T>
    {
        private PyObject _rawData;

        public PyArray<T> Grad
        {
            get { return _rawData["grad"]; }
            set { _rawData["grad"] = value; }
        }

        public PyArray<T> Data
        {
            get { return _rawData["data"]; }
            set { _rawData["data"] = value; }
        }

        public PyArray<T> Shape
        {
            get { return _rawData["shape"]; }
        }

        public Variable(PyArray<T> array)
        {
            _rawData = Chainer.Variable.Call(array);
        }

        public void Backward()
        {
            _rawData["backward"].Call();
        }

        public static PyObject operator +(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(Variable<T> x, Variable<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }


        //
        public static PyObject operator +(PyObject x, Variable<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(PyObject x, Variable<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(PyObject x, Variable<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(PyObject x, Variable<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        //
        public static PyObject operator +(Variable<T> x, PyObject y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(Variable<T> x, PyObject y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(Variable<T> x, PyObject y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(Variable<T> x, PyObject y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        public static implicit operator PyObject(Variable<T> variable)
        {
            return Unsafe.As<Variable<T>, PyObject>(ref variable);
        }

        public static implicit operator Variable<T>(PyObject pyObject)
        {
            PyObject result = Python.GetNamelessObject(pyObject);
            return Unsafe.As<PyObject, Variable<T>>(ref result);
        }
    }
}