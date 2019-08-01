using System;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "Linear", Namespace = "KelpNet")]
    public class Linear : SingleInputFunction, ICompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        [DataMember]
        public NdArray Weight;

        [DataMember]
        public NdArray Bias;


        [DataMember]
        public bool NoBias;


        [DataMember]
        public int InputCount;

        [DataMember]
        public int OutputCount;


        [DataMember]
        public ICompressibleActivation Activation { get; set; }

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.OutputCount = outputCount;
            this.InputCount = inputCount;

            this.Weight = new NdArray(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.Activation = activation;

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!noBias)
            {
                this.Bias = new NdArray(outputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.ToRealArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        public Linear(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(this.Bias.Data, 0, y, i * this.OutputCount, this.Bias.Data.Length);
            }

            return y;
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] y = this.NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    for (int j = 0; j < this.InputCount; j++)
                    {
                        y[batchCount * this.OutputCount + i] += x.Data[batchCount * this.InputCount + j] * this.Weight.Data[i * this.InputCount + j];
                    }
                }
            }

            if (this.Activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = this.Activation.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        protected void CalcBiasGrad(Real[] gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    this.Bias.Grad[i] += gy[batchCounter * this.OutputCount + i];
                }
            }
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activation != null ? this.GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * this.OutputCount];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.Weight.Grad[i * this.InputCount + j] += x.Data[batchCount * this.InputCount + j] * gyData;
                        x.Grad[batchCount * this.InputCount + j] += this.Weight.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }
        }

        public virtual Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
