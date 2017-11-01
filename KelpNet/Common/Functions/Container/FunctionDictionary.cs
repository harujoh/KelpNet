using System;
using System.Collections.Generic;
using KelpNet.Common.Functions.Type;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions.Container
{
    [Serializable]
    public class FunctionDictionary : Function
    {
        const string FUNCTION_NAME = "FunctionDictionary";

        //関数に入出力のキーを付加したFunctionRecordという単位で管理する
        public Dictionary<string, FunctionRecord> FunctionBlockDictionary = new Dictionary<string, FunctionRecord>();

        //辞書の実行順リスト
        public List<FunctionRecord> FunctionBlocks = new List<FunctionRecord>();

        //コンストラクタ
        public FunctionDictionary(string name = FUNCTION_NAME) : base(name)
        {
        }

        //関数の追加
        public void Add(string functionBlockName, Function function, string[] inputNames, string[] outputNames, bool compress = true)
        {            
            if (compress && //分岐毎の纏めを行うか
                (function is SingleInputFunction || function is MultiOutputFunction)) //まとめられる対象は入力が一つの関数のみ
            {
                //入力が登録済みの場合連結する
                if (FunctionBlockDictionary.ContainsKey(inputNames[0]))
                {
                    FunctionRecord functionBlock = this.FunctionBlockDictionary[inputNames[0]];
                    functionBlock.Add(function);

                    //出力名称を上書き
                    functionBlock.OutputNames = outputNames;

                    //リンクを追加
                    if (!(function is MultiOutputFunction) && //分岐する場合は親を登録せずリンクを切る
                        !FunctionBlockDictionary.ContainsKey(outputNames[0]))
                    {
                        FunctionBlockDictionary.Add(outputNames[0], functionBlock);
                    }

                    return;
                }
            }

            //既に辞書登録されている
            if (FunctionBlockDictionary.ContainsKey(functionBlockName))
            {
                FunctionBlockDictionary[functionBlockName].Add(function);
            }
            else
            {
                //どこにも登録がない
                //ブロックを新規に作る
                FunctionRecord functionRecord = new FunctionRecord(function, inputNames, outputNames, functionBlockName);
                FunctionBlocks.Add(functionRecord);
                FunctionBlockDictionary.Add(function.Name, functionRecord);
            }
        }

        //Forward
        public override NdArray[] Forward(params NdArray[] xs)
        {
            NdArray[] result = xs;

            //出力データの辞書
            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            //最初のデータを辞書に登録
            for (int i = 0; i < FunctionBlocks[0].InputNames.Length; i++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[i], xs[i]);
            }

            //登録順で実行していく
            for (int i = 0; i < FunctionBlocks.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[i].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(outPuts[inputBlockNames[j]]);
                }

                //関数を実施
                result = FunctionBlocks[i].Forward(inputData.ToArray());

                //出力したデータを辞書に登録
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(FunctionBlocks[i].OutputNames[j], result[j]);
                }
            }

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
                functionBlock.Update();
            }
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public override void ResetState()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.ResetState();
            }
        }

        //予想を実行する
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] result = xs;

            //出力データの辞書
            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            //出力したデータを辞書に登録
            for (int j = 0; j < FunctionBlocks[0].InputNames.Length; j++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[j], xs[j]);
            }

            for (int i = 0; i < FunctionBlocks.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[i].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(outPuts[inputBlockNames[j]]);
                }

                //関数を実施
                result = FunctionBlocks[i].Predict(inputData.ToArray());

                //出力したデータを辞書に登録
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(FunctionBlocks[i].OutputNames[j], result[j]);
                }
            }

            return result;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.SetOptimizer(optimizers);
            }
        }
    }
}
