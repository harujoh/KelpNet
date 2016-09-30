using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace VocabularyMaker
{
    public class Vocabulary
    {
        public List<string> Data = new List<string>();

        public int Length
        {
            get { return this.Data.Count; }
        }

        public int[] LoadData(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            StreamReader sr = new StreamReader(fs);
            string strText = sr.ReadToEnd();
            sr.Close();

            var replace = strText.Replace("\n", "<EOS>").Trim().Split();

            //ダブリを除いて辞書に追加
            this.Data.AddRange(replace);

            this.Data = new List<string>(this.Data.Distinct());

            int[] result = new int[replace.Length];
            for (int i = 0; i < replace.Length; i++)
            {
                result[i] = this.Data.IndexOf(replace[i]);
            }

            return result;
        }
    }
}
