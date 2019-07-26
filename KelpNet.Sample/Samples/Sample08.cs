using System;
using System.Collections.Generic;

namespace KelpNet.Sample.Samples
{
    //LSTMによるSin関数の学習（t の値から t+1 の値を予測する）
    //参考： http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
    class Sample08<T> where T : unmanaged, IComparable<T>
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
            NdArray<T> trainData = dataMaker.Make();

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack<T> model = new FunctionStack<T>(
                new Linear<T>(1, 5, name: "Linear l1"),
                new LSTM<T>(5, 5, name: "LSTM l2"),
                new Linear<T>(5, 1, name: "Linear l3")
            );

            //optimizerを宣言
            model.SetOptimizer(new Adam<T>());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray<T>[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

                Real<T> loss = ComputeLoss(model, sequences);

                model.Update();

                model.ResetState();

                if (epoch != 0 && epoch % DISPLAY_EPOCH == 0)
                {
                    Console.WriteLine("[{0}]training loss:\t{1}", epoch, loss);
                }
            }

            Console.WriteLine("Testing...");
            NdArray<T>[] testSequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

            int sample_index = 45;
            Predict(testSequences[sample_index], model, PREDICTION_LENGTH);
        }

        static Real<T> ComputeLoss(FunctionStack<T> model, NdArray<T>[] sequences)
        {
            //全体での誤差を集計
            Real<T> totalLoss = 0;
            NdArray<T> x = new NdArray<T>(new[] { 1 }, MINI_BATCH_SIZE);
            NdArray<T> t = new NdArray<T>(new[] { 1 }, MINI_BATCH_SIZE);

            Stack<NdArray<T>[]> backNdArrays = new Stack<NdArray<T>[]>();

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    x.Data[j] = sequences[j].Data[i];
                    t.Data[j] = sequences[j].Data[i + 1];
                }

                NdArray<T>[] result = model.Forward(x);
                totalLoss += new MeanSquaredError<T>().Evaluate(result, t);
                backNdArrays.Push(result);
            }

            for (int i = 0; backNdArrays.Count > 0; i++)
            {
                model.Backward(backNdArrays.Pop());
            }

            return totalLoss / (LENGTH_OF_SEQUENCE - 1);
        }

        static void Predict(NdArray<T> seq, FunctionStack<T> model, int pre_length)
        {
            RealArray<T> pre_input_seq = new T[seq.Data.Length / 4];
            if (pre_input_seq.Length < 1)
            {
                pre_input_seq = new T[1];
            }
            //Array.Copy(seq.Data, pre_input_seq, pre_input_seq.Length);
            seq.Data.CopyTo(pre_input_seq, 0, 0, pre_input_seq.Length);

            List<Real<T>> input_seq = new List<Real<T>>();
            input_seq.AddRange(pre_input_seq);

            List<Real<T>> output_seq = new List<Real<T>>();
            output_seq.Add(input_seq[input_seq.Count - 1]);

            for (int i = 0; i < pre_length; i++)
            {
                Real<T> future = Predict_sequence(model, input_seq);
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

        private static Real<T> Predict_sequence(FunctionStack<T> model, List<Real<T>> input_seq)
        {
            model.ResetState();

            NdArray<T> result = new NdArray<T>(1);

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

            public NdArray<T> Make()
            {
                NdArray<T> result = new NdArray<T>(this.stepsPerCycle * this.numberOfCycles);

                for (int i = 0; i < this.numberOfCycles; i++)
                {
                    for (int j = 0; j < this.stepsPerCycle; j++)
                    {
                        result.Data[j + i * this.stepsPerCycle] = Math.Sin(j * 2 * Math.PI / this.stepsPerCycle);
                    }
                }

                return result;
            }

            public NdArray<T>[] MakeMiniBatch(NdArray<T> baseFreq, int miniBatchSize, int lengthOfSequence)
            {
                NdArray<T>[] result = new NdArray<T>[miniBatchSize];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray<T>(lengthOfSequence);

                    int index = Mother<T>.Dice.Next(baseFreq.Data.Length - lengthOfSequence);
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
