using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function<T> : IFunction<T> where T : unmanaged, IComparable<T>
    {
        public string Name { get; set; }
        public string[] InputNames { get; set; }
        public string[] OutputNames { get; set; }

        public NdArray<T>[] Parameters { get; set; } = { };

        [field: NonSerialized]
        public List<NdArray<T>[]> PrevInputs { get; set; } = new List<NdArray<T>[]>();

        [field: NonSerialized]
        public List<NdArray<T>[]> UsedPrevInputs { get; set; } = new List<NdArray<T>[]>();

        [field: NonSerialized]
        public FunctionOptional<T> Forward { get; set; }

        [field: NonSerialized]
        public FunctionOptional<T> Predict { get; set; }

        [field: NonSerialized]
        public ActionOptional<T> Backward { get; set; }

        //コンストラクタ
        protected Function(string name, string[] inputNames = null, string[] outputNames = null)
        {
            this.Name = name;

            if (inputNames != null)
            {
                this.InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                this.OutputNames = outputNames.ToArray();
            }
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            PrevInputs = new List<NdArray<T>[]>();
            UsedPrevInputs = new List<NdArray<T>[]>();
        }

        //パラメータを更新する時に呼ぶ関数
        protected void BackwardCountUp()
        {
            foreach (NdArray<T> parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        public void InitGrad()
        {
            foreach (NdArray<T> parameter in this.Parameters)
            {
                if (parameter.Grad == null) parameter.InitGrad();
            }
        }

        //RNN等で使い切れなかった入力データを初期化
        public virtual void ResetState()
        {
            this.PrevInputs = new List<NdArray<T>[]>();
            this.UsedPrevInputs = new List<NdArray<T>[]>();
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }
    }
}
