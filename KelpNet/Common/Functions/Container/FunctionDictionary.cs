using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.CPU;

namespace KelpNet
{
    [Serializable]
    public class FunctionDictionary : Function
    {
        const string FUNCTION_NAME = "FunctionDictionary";

        //関数に入出力のキーを付加したFunctionRecordという単位で管理する
        public Dictionary<string, FunctionStack> FunctionBlockDictionary = new Dictionary<string, FunctionStack>();

        //分割関数の名前を保持する辞書
        public Dictionary<string, FunctionStack> SplitedFunctionDictionary = new Dictionary<string, FunctionStack>();

        //辞書の実行順リスト
        public List<FunctionStack> FunctionBlocks = new List<FunctionStack>();

        private readonly bool _compress = false;

        //コンストラクタ
        public FunctionDictionary(bool compress = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._compress = compress;
        }

        //関数の追加
        public void Add(Function function)
        {
            if (_compress && //分岐毎のまとめを行うか
                (function is SingleInputFunction || function is MultiOutputFunction)) //入力が一つの関数のみまとめられる
            {
                //入力名称で辞書に登録が有るかチェック
                if (this.FunctionBlockDictionary.ContainsKey(function.InputNames[0]))
                {
                    //ブロックが既に辞書登録されていればブロックに連結
                    this.FunctionBlockDictionary[function.InputNames[0]].Add(function);

                    //出力名称を上書き
                    this.FunctionBlockDictionary[function.InputNames[0]].OutputNames = function.OutputNames.ToArray();

                    //分割済み関数なら分割元の出力名を更新する
                    if (SplitedFunctionDictionary.ContainsKey(function.InputNames[0]))
                    {
                        FunctionStack spliteFunction = SplitedFunctionDictionary[function.InputNames[0]];

                        for (int i = 0; i < spliteFunction.OutputNames.Length; i++)
                        {
                            if (spliteFunction.OutputNames[i] == function.InputNames[0])
                            {
                                spliteFunction.OutputNames[i] = function.OutputNames[0];

                                if (!SplitedFunctionDictionary.ContainsKey(function.OutputNames[0]))
                                {
                                    SplitedFunctionDictionary.Add(function.OutputNames[0], spliteFunction);
                                }
                            }
                        }
                    }

                    if (!(function is MultiOutputFunction) && //出力が分岐する場合は登録せずリンクを切る
                      !this.FunctionBlockDictionary.ContainsKey(function.OutputNames[0])) //既に登録されている場合は登録しない
                    {
                        //リンクを辞書へ追加
                        this.FunctionBlockDictionary.Add(function.OutputNames[0], this.FunctionBlockDictionary[function.InputNames[0]]);
                    }
                    else if (function is SplitFunction) //SplitFunctionの場合
                    {
                        var splitFunctions = ((SplitFunction)function).SplitedFunctions;

                        for (int i = 0; i < splitFunctions.Length; i++)
                        {
                            //内部のFunctionStackをリンクの辞書へ追加
                            FunctionBlockDictionary.Add(function.OutputNames[i], splitFunctions[i]);

                            //SplitFunctionのリストへ追加
                            SplitedFunctionDictionary.Add(function.OutputNames[i], this.FunctionBlockDictionary[function.InputNames[0]]);
                        }
                    }

                    return;
                }
            }

            //以下無圧縮、もしくはMultiInput,DualInput用の処理

            //ブロックが辞書に登録されているか
            if (this.FunctionBlockDictionary.ContainsKey(function.OutputNames[0]))
            {
                //ブロックが既に辞書登録されていればブロックに連結
                this.FunctionBlockDictionary[function.OutputNames[0]].Add(function);
            }
            else
            {
                //登録がなければブロックを新規に作る
                FunctionStack functionRecord = new FunctionStack(function, function.Name, function.InputNames, function.OutputNames);

                //実行順に登録
                this.FunctionBlocks.Add(functionRecord);

                //リンクを辞書へ追加
                this.FunctionBlockDictionary.Add(function.Name, functionRecord);
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
                NdArray[] inputData = new NdArray[inputBlockNames.Length];

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData[j] = outPuts[inputBlockNames[j]];
                }

                //関数を実施
                result = FunctionBlocks[i].Forward(inputData);

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

            ResetState();
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
                NdArray[] inputData = new NdArray[inputBlockNames.Length];

                //入力するデータを集めてくる
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData[j] = outPuts[inputBlockNames[j]];
                }

                //関数を実施
                result = FunctionBlocks[i].Predict(inputData);

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
