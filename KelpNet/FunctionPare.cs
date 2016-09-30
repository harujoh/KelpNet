using System;
using System.Collections.Generic;
using KelpNet.Interface;

namespace KelpNet
{
    //ファンクションペア（非バッチ処理をまとめ、極力複数層に渡った並列処理を実現する
    [Serializable]
    public class FunctionPare
    {
        public List<IBatchFunction> BatchFunctions = new List<IBatchFunction>();
        public List<Function> SoloFunctions = new List<Function>();
    }
}
