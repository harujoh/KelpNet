using System;
using System.Collections.Generic;
using KelpNet.CL;

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
            NdArray trainData = dataMaker.Make();

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack model = new FunctionStack(
                new Linear(1, 5, name: "Linear l1"),
                new LSTM(5, 5, name: "LSTM l2"),
                new Linear(5, 1, name: "Linear l3")
            );

            //optimizerを宣言
            model.SetOptimizer(new Adam());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

                Real loss = ComputeLoss(model, sequences);

                model.Update();

                model.ResetState();

                if (epoch != 0 && epoch % DISPLAY_EPOCH == 0)
                {
                    Console.WriteLine("[{0}]training loss:\t{1}", epoch, loss);
                }
            }

            Console.WriteLine("Testing...");
            NdArray[] testSequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

            int sample_index = 45;
            predict(testSequences[sample_index], model, PREDICTION_LENGTH);
        }

        static Real ComputeLoss(FunctionStack model, NdArray[] sequences)
        {
            //全体での誤差を集計
            Real totalLoss = 0;
            NdArray x = new NdArray(new[] { 1 }, MINI_BATCH_SIZE);
            NdArray t = new NdArray(new[] { 1 }, MINI_BATCH_SIZE);

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x.Data[j] = sequences[j].Data[i];
                    t.Data[j] = sequences[j].Data[i + 1];
                }

                NdArray result = model.Forward(x)[0];
                totalLoss += new MeanSquaredError().Evaluate(result, t);
                model.Backward(result);
            }

            return totalLoss / (LENGTH_OF_SEQUENCE - 1);
        }

        static void predict(NdArray seq, FunctionStack model, int pre_length)
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

        static Real predict_sequence(FunctionStack model, List<Real> input_seq)
        {
            model.ResetState();

            NdArray result = 0;

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

            public NdArray Make()
            {
                NdArray result = new NdArray(this.stepsPerCycle * this.numberOfCycles);

                for (int i = 0; i < this.numberOfCycles; i++)
                {
                    for (int j = 0; j < this.stepsPerCycle; j++)
                    {
                        result.Data[j + i * this.stepsPerCycle] = Math.Sin(j * 2 * Math.PI / this.stepsPerCycle);
                    }
                }

                return result;
            }

            public NdArray[] MakeMiniBatch(NdArray baseFreq, int miniBatchSize, int lengthOfSequence)
            {
                NdArray[] result = new NdArray[miniBatchSize];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray(lengthOfSequence);

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
