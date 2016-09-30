using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //LSTMによる正弦波の予測（t の値から t+1 の値を予測する）
    //参考： http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
    class Test8
    {
        const int STEPS_PER_CYCLE = 50;
        const int NUMBER_OF_CYCLES = 100;

#if DEBUG
        const int TRAINING_EPOCHS = 100;
        const int MINI_BATCH_SIZE = 1;
        const int LENGTH_OF_SEQUENCE = 5;
#else
        const int TRAINING_EPOCHS = 1000;
        const int MINI_BATCH_SIZE = 100;
        const int LENGTH_OF_SEQUENCE = 100;
#endif

        const int DISPLAY_EPOCH = 2;
        const int PREDICTION_LENGTH = 75;

        public static void Run()
        {
            var dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
            var trainData = dataMaker.Make();

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack model = new FunctionStack(
                new Linear(1, 5,
#if DEBUG
                    initialW: new[,] { { 0.06301765 }, { 0.02956826 }, { -0.0451562 }, { 0.08234128 }, { -0.02225312 } },
                    initialb: new[] { 0.04880815, -0.09647872, 0.04342706, 0.00905941, -0.08944918 },
#endif
                    name: "Linear l1"),
                new LSTM(5, 5, name: "LSTM l2"),
                new Linear(5, 1,
#if DEBUG
                    initialW: new[,] { { 0.03912614, -0.01009424, -0.05674245, 0.07616881, 0.0120074 } },
                    initialb: new[] { -0.04287718 },
#endif
                    name: "Linear l3")
            );

            //optimizerを宣言
            model.SetOptimizer(new Adam());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
#if DEBUG
                var sequences = new[] { NdArray.FromArray(new[] { 0.58778524, 0.68454713, 0.77051324, 0.84432792, 0.90482705 }) };
#else
                var sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);
#endif
                model.ResetState();
                model.ClearGrads();

                var loss = ComputeLoss(model, sequences);

                model.Update();

                if (epoch != 0 && epoch % DISPLAY_EPOCH == 0)
                {
                    Console.WriteLine("[{0}]training loss:\t{1}", epoch, loss);
                }
            }

            Console.WriteLine("Testing...");
            var testSequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

#if DEBUG
            int sample_index = 0;
#else
            int sample_index = 45;
#endif
            predict(testSequences[sample_index], model, PREDICTION_LENGTH);
        }

        static double ComputeLoss(FunctionStack model, NdArray[] sequences)
        {
            //全体での誤差を集計
            List<double> totalLoss = new List<double>();
            var x = new double[MINI_BATCH_SIZE][];
            var t = new double[MINI_BATCH_SIZE][];

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            //入出力を初期化
            model.InitBatch(MINI_BATCH_SIZE);

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x[j] = new[] { sequences[j].Data[i] };
                    t[j] = new[] { sequences[j].Data[i + 1] };
                }

                List<double> sumLoss;
                backNdArrays.Push(model.BatchForward(x, t, LossFunctions.MeanSquaredError, out sumLoss));
                totalLoss.AddRange(sumLoss);
            }

            for (int i = 0; backNdArrays.Count > 0; i++)
            {
                model.BatchBackward(backNdArrays.Pop());
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
            Array.Copy(seq.Data, pre_input_seq, pre_input_seq.Length);

            List<double> input_seq = new List<double>();
            input_seq.AddRange(pre_input_seq);

            List<double> output_seq = new List<double>();
            output_seq.Add(input_seq[input_seq.Count - 1]);

            for (int i = 0; i < pre_length; i++)
            {
                var future = predict_sequence(model, input_seq);
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
            
            NdArray result = NdArray.Empty(1);

            for (int i = 0; i < input_seq.Count; i++)
            {
                result = model.Predict(new[]{input_seq[i]});
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
                NdArray result = NdArray.Empty(this.stepsPerCycle * this.numberOfCycles);

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
                    result[j] = NdArray.Empty(lengthOfSequence);

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
