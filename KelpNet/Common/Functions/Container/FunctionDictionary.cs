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
        public Dictionary<string, FunctionStack> FunctionBlockDictionary = new Dictionary<string, FunctionStack>();

        //辞書の実行順リスト
        public List<FunctionStack> FunctionBlocks = new List<FunctionStack>();

        private readonly bool _compress = false;

        //コンストラクタ
        public FunctionDictionary(bool compress = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._compress = compress;
        }

        //関数の追加
        public void Add(Function function, string functionBlockName)
        {
            if (_compress && //分岐毎のまとめを行うか
                (function is SingleInputFunction || function is MultiOutputFunction)) //入力が一つの関数のみまとめられる
            {
                //入力名称で辞書に登録が有るかチェック
                if (FunctionBlockDictionary.ContainsKey(function.InputNames[0]))
                {
                    //入力が登録済みの場合連結する
                    FunctionStack functionBlock = this.FunctionBlockDictionary[function.InputNames[0]];
                    functionBlock.Add(function);

                    //出力名称を上書き
                    functionBlock.OutputNames = function.OutputNames;

                    if (!(function is MultiOutputFunction) && //出力が分岐する場合は登録せずリンクを切る
                        !FunctionBlockDictionary.ContainsKey(function.OutputNames[0]))//既に登録されている場合は登録しない
                    {
                        //リンクを追加
                        FunctionBlockDictionary.Add(function.OutputNames[0], functionBlock);
                    }

                    return;
                }
            }

            //ブロックが辞書に登録されているか
            if (FunctionBlockDictionary.ContainsKey(functionBlockName))
            {
                //ブロックが既に辞書登録されていれば追加
                FunctionBlockDictionary[functionBlockName].Add(function);
            }
            else
            {
                //登録がなければブロックを新規に作る
                FunctionStack functionRecord = new FunctionStack(function, functionBlockName, function.InputNames, function.OutputNames);
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
        public override void Backward(params NdArray[] ys)
        {
            NdArray.Backward(ys[0]);
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
