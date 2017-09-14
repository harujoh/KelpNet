using System.Windows.Forms;
using CaffemodelLoader;

namespace KelpNetTester.Tests
{
    //CaffemodelLoadテスト
    class Test15
    {
        public static void Run()
        {
            OpenFileDialog ofd = new OpenFileDialog();

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                var test = CaffemodelDataLoader.ModelLoad(ofd.FileName);
            }
        }
    }
}
