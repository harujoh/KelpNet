using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace KelpNet.Sample
{
    public class Vocabulary
    {
        public List<string> Data = new List<string>();
        public int EosID = -1;

        public int Length
        {
            get { return this.Data.Count; }
        }

        public int[] LoadData(string fileName)
        {
            int[] result;

            using (FileStream fs = new FileStream(fileName, FileMode.Open))
            {
                StreamReader sr = new StreamReader(fs);
                string strText = sr.ReadToEnd();
                sr.Close();

                string[] replace = strText.Replace("\r\n", "\n").Replace("\n", "<EOS>").Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                //ダブリを除いて辞書に追加
                this.Data.AddRange(replace);

                this.Data = new List<string>(this.Data.Distinct());

                result = new int[replace.Length];
                for (int i = 0; i < replace.Length; i++)
                {
                    result[i] = this.Data.IndexOf(replace[i]);
                }

                if (this.EosID == -1)
                {
                    this.EosID = this.Data.IndexOf("<EOS>");
                }
            }

            return result;
        }
    }
}
