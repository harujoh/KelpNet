using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace CaffemodelLoader
{
    [Serializable]
    public class FunctionDictionary : Function
    {
        const string FUNCTION_NAME = "FunctionDictionary";

        //出力名称のKey付きのファンクションブロック
        public Dictionary<string, FunctionRecord> FunctionBlocks = new Dictionary<string, FunctionRecord>();

        //出力データの辞書
        public Dictionary<string, NdArray> OutPuts = new Dictionary<string, NdArray>();

        //辞書の実行順リスト
        public List<string> FunctionBlockNames = new List<string>();

        //コンストラクタ
        public FunctionDictionary(string name = FUNCTION_NAME) : base(name)
        {
        }

        //関数の追加
        public void Add(string functionBlockName, Function function, string[] inputNames, string[] outputNames)
        {
            //既にブロックがあればそこへ追加
            if (FunctionBlocks.ContainsKey(outputNames[0]))
            {
                //既にブロックがあればそこへ追加
                FunctionBlocks[outputNames[0]].Add(function);
            }
            else
            {
                //ブロックを新規に作る
                FunctionBlockNames.Add(functionBlockName);
                FunctionRecord functionRecord = new FunctionRecord(function, inputNames, outputNames, functionBlockName);
                FunctionBlocks.Add(functionBlockName, functionRecord);
            }
        }

        //Forward
        public override NdArray[] Forward(params NdArray[] xs)
        {
            NdArray[] result = FunctionBlocks[FunctionBlockNames[0]].Forward(xs);

            //出力したデータを辞書に登録
            for (int j = 0; j < result.Length; j++)
            {
                OutPuts.Add(FunctionBlocks[FunctionBlockNames[0]].OutputNames[j], result[j]);
            }

            for (int i = 1; i < FunctionBlockNames.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[FunctionBlockNames[i]].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(OutPuts[inputBlockNames[j]]);
                }

                //関数を実施
                result = FunctionBlocks[FunctionBlockNames[i]].Forward(inputData.ToArray());

                //出力したデータを辞書に登録
                for (int j = 0; j < result.Length; j++)
                {
                    OutPuts.Add(FunctionBlocks[FunctionBlockNames[i]].OutputNames[j],result[j]);
                }
            }

            OutPuts.Clear();

            return result;
        }

        //Backward
        public override void Backward(params NdArray[] y)
        {
            NdArray.Backward(y[0]);
        }

        //重みの更新処理
        public override void Update()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.Update();
            }
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public override void ResetState()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.ResetState();
            }
        }

        //予想を実行する
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] result = FunctionBlocks[FunctionBlockNames[0]].Predict(xs);
            //出力したデータを辞書に登録
            for (int j = 0; j < result.Length; j++)
            {
                OutPuts.Add(FunctionBlocks[FunctionBlockNames[0]].OutputNames[j], result[j]);
            }

            for (int i = 1; i < FunctionBlockNames.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[FunctionBlockNames[i]].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(OutPuts[inputBlockNames[j]]);
                }

                //関数を実施
                result = FunctionBlocks[FunctionBlockNames[i]].Predict(inputData.ToArray());

                //出力したデータを辞書に登録
                for (int j = 0; j < result.Length; j++)
                {
                    OutPuts.Add(FunctionBlocks[FunctionBlockNames[i]].OutputNames[j], result[j]);
                }
            }

            OutPuts.Clear();

            return result;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.SetOptimizer(optimizers);
            }
        }
    }
}
