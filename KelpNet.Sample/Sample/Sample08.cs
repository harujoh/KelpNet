using System;
using System.Collections.Generic;
using KelpNet.CL;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#else
using KelpMath = KelpNet.MathF;
#endif

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    //LSTMによるSin関数の学習（t の値から t+1 の値を予測する）
    //参考： http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
    class Sample08
    {
        const int STEPS_PER_CYCLE = 50;
        const int NUMBER_OF_CYCLES = 100;

        const int TRAINING_EPOCHS = 1000;
        const int MINI_BATCH_SIZE = 100;
        const int LENGTH_OF_SEQUENCE = 100;

        const int DISPLAY_EPOCH = 1;
        const int PREDICTION_LENGTH = 75;

        public static void Run()
        {
            DataMaker dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
            NdArray<Real> trainData = dataMaker.Make();

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack<Real> model = new FunctionStack<Real>(
                new Linear<Real>(1, 5, name: "Linear l1"),
                new LSTM<Real>(5, 5, name: "LSTM l2"),
                new Linear<Real>(5, 1, name: "Linear l3")
            );

            //optimizerを宣言
            Adam<Real> adam = new Adam<Real>();
            adam.SetUp(model);

            //訓練ループ
            Console.WriteLine("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray<Real>[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

                Real loss = ComputeLoss(model, sequences);

                adam.Update();

                model.ResetState();

                if (epoch != 0 && epoch % DISPLAY_EPOCH == 0)
                {
                    Console.WriteLine("[{0}]training loss:\t{1}", epoch, loss);
                }
            }

            Console.WriteLine("Testing...");
            NdArray<Real>[] testSequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

            int sample_index = 45;
            predict(testSequences[sample_index], model, PREDICTION_LENGTH);
        }

        static Real ComputeLoss(FunctionStack<Real> model, NdArray<Real>[] sequences)
        {
            //全体での誤差を集計
            Real totalLoss = 0;
            NdArray<Real> x = new NdArray<Real>(new[] { 1 }, MINI_BATCH_SIZE);
            NdArray<Real> t = new NdArray<Real>(new[] { 1 }, MINI_BATCH_SIZE);

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x.Data[j] = sequences[j].Data[i];
                    t.Data[j] = sequences[j].Data[i + 1];
                }

                NdArray<Real> result = model.Forward(x)[0];
                totalLoss += new MeanSquaredError<Real>().Evaluate(result,  t);
                model.Backward(result);
            }

            return totalLoss / (LENGTH_OF_SEQUENCE - 1);
        }

        static void predict(NdArray<Real> seq, FunctionStack<Real> model, int pre_length)
        {
            Real[] pre_input_seq = new Real[seq.Data.Length / 4];
            if (pre_input_seq.Length < 1)
            {
                pre_input_seq = new Real[1];
            }
            Array.Copy(seq.Data, pre_input_seq, pre_input_seq.Length);

            List<Real> input_seq = new List<Real>();
            input_seq.AddRange(pre_input_seq);

            List<Real> output_seq = new List<Real>();
            output_seq.Add(input_seq[input_seq.Count - 1]);

            for (int i = 0; i < pre_length; i++)
            {
                Real future = predict_sequence(model, input_seq);
                input_seq.RemoveAt(0);
                input_seq.Add(future);
                output_seq.Add(future);
            }

            for (int i = 0; i < output_seq.Count; i++)
            {
                Console.WriteLine(output_seq[i]);
            }

            Console.WriteLine(seq);
        }

        static Real predict_sequence(FunctionStack<Real> model, List<Real> input_seq)
        {
            model.ResetState();

            NdArray<Real> result = 0;

            for (int i = 0; i < input_seq.Count; i++)
            {
                result = model.Predict(input_seq[i])[0];
            }

            return result.Data[0];
        }

        class DataMaker
        {
            private readonly int stepsPerCycle;
            private readonly int numberOfCycles;

            public DataMaker(int stepsPerCycle, int numberOfCycles)
            {
                this.stepsPerCycle = stepsPerCycle;
                this.numberOfCycles = numberOfCycles;
            }

            public NdArray<Real> Make()
            {
                NdArray<Real> result = new NdArray<Real>(this.stepsPerCycle * this.numberOfCycles);

                for (int i = 0; i < this.numberOfCycles; i++)
                {
                    for (int j = 0; j < this.stepsPerCycle; j++)
                    {
                        result.Data[j + i * this.stepsPerCycle] = KelpMath.Sin(j * 2 * KelpMath.PI / this.stepsPerCycle);
                    }
                }

                return result;
            }

            public NdArray<Real>[] MakeMiniBatch(NdArray<Real> baseFreq, int miniBatchSize, int lengthOfSequence)
            {
                NdArray<Real>[] result = new NdArray<Real>[miniBatchSize];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray<Real>(lengthOfSequence);

                    int index = Mother.Dice.Next(baseFreq.Data.Length - lengthOfSequence);
                    for (int j = 0; j < lengthOfSequence; j++)
                    {
                        result[i].Data[j] = baseFreq.Data[index + j];
                    }

                }

                return result;
            }
        }
    }
}
