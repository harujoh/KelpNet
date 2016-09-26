namespace KelpNet
{
    public class OptimizeParameter
    {
        public string Name;
        public NdArray Param;
        public NdArray Grad;

        //Updateを行わずに実行されたTrainの回数をカウントし、バッチ更新時に使用する
        public int TrainCount;

        public int Length
        {
            get
            {
                return this.Param.Length;
            }
        }

        public OptimizeParameter(NdArray param, NdArray grad, string name)
        {
            this.Param = param;
            this.Grad = grad;
            this.Name = name;
        }

        //傾きの初期化
        public void ClearGrad()
        {
            this.Grad.Fill(0);
            //バッチカウントもリセット
            this.TrainCount = 0;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}
