using System;
using System.Runtime.CompilerServices;
using NConstrictor;

namespace ChainerCore
{
    class Chainer
    {
        private static PyObject _chainer;
        public static PyObject Variable;
        public static PyObject Functions;
        public static PyObject Links;
        public static PyObject Optimizers;

        public static void Initialize()
        {
            _chainer = PyImport.ImportModule("chainer");

            //todo debug
            Python.Run("import chainer");

            if (_chainer == IntPtr.Zero)
            {
                throw new Exception("chainer failed to import");
            }

            Variable = _chainer["Variable"];
            Links = _chainer["links"];
            Functions = _chainer["functions"];
            Optimizers = _chainer["optimizers"];
        }

        public static PyObject Linear(PyObject in_size, PyObject out_size)
        {
            return Links["Linear"].Call(in_size, out_size);
        }

        public static PyObject Grad(PyObject outputs, PyObject inputs)
        {
            //outputs, inputs, grad_outputs=None, grad_inputs=None, set_grad=False, retain_grad=False, enable_double_backprop=False, loss_scale=None
            return _chainer["grad"].Call(outputs, inputs);
        }
    }

    public struct Variable<T>
    {
        private PyArray<T> _rawData;

        //todo debug
        public Variable(PyObject array)
        {
            Py.IncRef(Python.Main);
            Python.Main["ax"] = array;

            Python.Run("x = chainer.Variable(ax)");

            Py.IncRef(Python.Main);
            _rawData = Python.Main["x"];
            //_rawData = Chainer.Variable.Call(array);
        }

        public static implicit operator PyArray<T>(Variable<T> i)
        {
            return Unsafe.As<Variable<T>, PyObject>(ref i);
        }

        public static implicit operator Variable<T>(PyArray<T> i)
        {
            return Unsafe.As<PyArray<T>, Variable<T>>(ref i);
        }

        public static implicit operator PyObject(Variable<T> i)
        {
            return Unsafe.As<Variable<T>, PyObject>(ref i);
        }

        public static implicit operator Variable<T>(PyObject i)
        {
            return Unsafe.As<PyObject, Variable<T>>(ref i);
        }
    }
}
