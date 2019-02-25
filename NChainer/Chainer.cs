using System;
using NConstrictor;

namespace NChainer
{
    public class Chainer
    {
        private static PyObject _chainer;
        public static PyObject Variable;
        public static PyObject Functions;
        public static PyObject Links;
        public static PyObject Optimizers;
        public static PyObject Config;

        public static void Initialize()
        {
            _chainer = PyImport.ImportModule("chainer");

            if (_chainer == IntPtr.Zero)
            {
                throw new Exception("chainer failed to import");
            }

            Variable = _chainer["Variable"];
            Links = _chainer["links"];
            Functions = _chainer["functions"];
            Optimizers = _chainer["optimizers"];
            Config = _chainer["config"]; 
        }
    }
}
