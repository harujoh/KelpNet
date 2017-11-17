using System.IO;

namespace TarArchive.Common
{
    public class Options
    {
        public TarCompression Compression;
        public TextWriter StatusWriter;
        public bool FollowSymLinks;
        public bool Overwrite;
        public bool DoNotSetTime;
    }
}
