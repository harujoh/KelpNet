using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //LSTMによるSin関数の学習（t の値から t+1 の値を予測する）
    //参考： http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
    class Test8
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
            Adam adam = new Adam(model.Parameters);

            //訓練ループ
            Console.WriteLine("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

                double loss = ComputeLoss(model, sequences);

                model.Update(adam);

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

        static double ComputeLoss(FunctionStack model, NdArray[] sequences)
        {
            //全体での誤差を集計
            List<double> totalLoss = new List<double>();
            double[][] x = new double[MINI_BATCH_SIZE][];
            double[][] t = new double[MINI_BATCH_SIZE][];

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x[j] = new[] { sequences[j].Data[i] };
                    t[j] = new[] { sequences[j].Data[i + 1] };
                }

                double sumLoss;
                backNdArrays.Push(Trainer.Forward(model, x, t, LossFunctions.MeanSquaredError, out sumLoss));
                totalLoss.Add(sumLoss);
            }

            for (int i = 0; backNdArrays.Count > 0; i++)
            {
                model.Backward(backNdArrays.Pop());
            }

            return totalLoss.Average();
        }

        static void predict(NdArray seq, FunctionStack model, int pre_length)
        {
            double[] pre_input_seq = new double[seq.Length / 4];
            if (pre_input_seq.Length < 1)
            {
                pre_input_seq = new double[1];
            }
            Buffer.BlockCopy(seq.Data, 0, pre_input_seq, 0, sizeof(double) * pre_input_seq.Length);

            List<double> input_seq = new List<double>();
            input_seq.AddRange(pre_input_seq);

            List<double> output_seq = new List<double>();
            output_seq.Add(input_seq[input_seq.Count - 1]);

            for (int i = 0; i < pre_length; i++)
            {
                double future = predict_sequence(model, input_seq);
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


        static double predict_sequence(FunctionStack model, List<double> input_seq)
        {
            model.ResetState();

            NdArray result = NdArray.Zeros(1);

            for (int i = 0; i < input_seq.Count; i++)
            {
                result = Trainer.Predict(model, new[] { input_seq[i] });
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
                NdArray result = NdArray.Zeros(this.stepsPerCycle * this.numberOfCycles);

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

                for (int j = 0; j < result.Length; j++)
                {
                    result[j] = NdArray.Zeros(lengthOfSequence);

                    int index = Mother.Dice.Next(baseFreq.Length - lengthOfSequence);
                    for (int i = 0; i < lengthOfSequence; i++)
                    {
                        result[j].Data[i] = baseFreq.Data[index + i];
                    }

                }

                return result;
            }
        }
    }
}
