using System.Runtime.CompilerServices;
using NConstrictor;

namespace NChainer
{
    public struct Variable
    {
        private PyObject _rawData;
        public PyObject Grad => _rawData["grad_var"];

        public Variable(PyObject array)
        {
            _rawData = Python.GetNamelessObject(Chainer.Variable.Call(array));
        }

        public void Backward()
        {
            _rawData["backward"].Call();
        }

        public static Variable operator +(Variable x, Variable y)
        {
            return PyNumber.Add(x, y);
        }

        public static Variable operator -(Variable x, Variable y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static Variable operator *(Variable x, Variable y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static Variable operator /(Variable x, Variable y)
        {
            return PyNumber.TrueDivide(x, y);
        }


        //
        public static Variable operator +(PyObject x, Variable y)
        {
            return PyNumber.Add(x, y);
        }

        public static Variable operator -(PyObject x, Variable y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static Variable operator *(PyObject x, Variable y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static Variable operator /(PyObject x, Variable y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        //
        public static Variable operator +(Variable x, PyObject y)
        {
            return PyNumber.Add(x, y);
        }

        public static Variable operator -(Variable x, PyObject y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static Variable operator *(Variable x, PyObject y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static Variable operator /(Variable x, PyObject y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        public static implicit operator PyObject(Variable variable)
        {
            return Unsafe.As<Variable, PyObject>(ref variable);
        }

        public static implicit operator Variable(PyObject pyObject)
        {
            return Unsafe.As<PyObject, Variable>(ref pyObject);
        }
    }
}
