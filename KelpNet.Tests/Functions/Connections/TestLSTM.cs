using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;

namespace KelpNet.Tests
{
    //[TestClass]
    public class TestLSTM
    {
        const int STEPS_PER_CYCLE = 50;
        const int NUMBER_OF_CYCLES = 100;

        const int MINI_BATCH_SIZE = 100;
        const int LENGTH_OF_SEQUENCE = 100;

        //[TestMethod]
        public void LSTMSinTest()
        {
            DataMaker dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
            NdArray trainData = dataMaker.Make();

            Real[,] upwardInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
            Real[,] lateralInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
            Real[] biasInit = { Mother.Dice.NextDouble() };
            Real[] forgetBiasInit = { Mother.Dice.NextDouble() };

            NChainer.LSTM<Real> clstm = new NChainer.LSTM<Real>(1, 1, Real.ToBaseNdArray(upwardInit), Real.ToBaseNdArray(lateralInit), Real.ToBaseArray(biasInit), Real.ToBaseArray(forgetBiasInit));

            KelpNet.LSTM lstm = new KelpNet.LSTM(1, 1, upwardInit, lateralInit, biasInit, forgetBiasInit);

            NdArray[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

            Real[] input = new Real[MINI_BATCH_SIZE];
            Real[] teach = new Real[MINI_BATCH_SIZE];

            Stack<NdArray> backNdArrays = new Stack<NdArray>();
            Stack<Variable<Real>> backVariables = new Stack<Variable<Real>>();

            for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
            {
                for (int j = 0; j < MINI_BATCH_SIZE; j++)
                {
                    input[j] = sequences[j].Data[i];
                    teach[j] = sequences[j].Data[i + 1];
                }

                Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));
                Variable<Real> cT = new Variable<Real>(Real.ToBaseNdArray(teach));
                Variable<Real> cY = clstm.Forward(cX);
                //Variable<Real> cLoss = new NChainer.MeanSquaredError<Real>().Forward(cY, cT);
                //backVariables.Push(cLoss);

                NdArray x = new NdArray(input, new[] { 1 }, MINI_BATCH_SIZE);
                NdArray t = new NdArray(teach, new[] { 1 }, MINI_BATCH_SIZE);
                NdArray y = lstm.Forward(x)[0];
                new KelpNet.MeanSquaredError().Evaluate(y, t);
                backNdArrays.Push(y);
            }

            for (int i = 0; backNdArrays.Count > 0; i++)
            {
                //backVariables.Pop().Backward();
                backNdArrays.Pop().Backward();
            }
        }
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
