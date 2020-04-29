﻿using System;
using System.Linq;
using System.Runtime.Serialization;

namespace KelpNet
{
    class Reshape<T> : SingleInputFunction<T> where T:unmanaged,IComparable<T>
    {
        const string FUNCTION_NAME = "Reshape";
        public int[] Shape;

        public Reshape(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Shape = shape;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.SingleInputForward = this.SingleInputForward;
            base.SingleOutputBackward = this.SingleOutputBackward;
        }

        public NdArray<T> SingleInputForward(NdArray<T> val)
        {
            NdArray<T> result = val.Clone();
            result.ParentFunc = this;
            result.Reshape(this.Shape);

            return result;
        }

        public void SingleOutputBackward(NdArray<T> y, NdArray<T> x)
        {
            y.Grad = x.Grad.ToArray();
        }
    }
    
}
