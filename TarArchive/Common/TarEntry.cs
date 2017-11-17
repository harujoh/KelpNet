using System;

namespace TarArchive.Common
{
    public class TarEntry
    {
        internal TarEntry() { }

        public string Name { get; internal set; }
        public int Size { get; internal set; }
        public DateTime Mtime { get; internal set; }
        public TarEntryType Type { get; internal set; }

        public char TypeChar
        {
            get
            {
                switch (Type)
                {
                    case TarEntryType.File_Old:
                    case TarEntryType.File:
                    case TarEntryType.File_Contiguous:
                        return 'f';
                    case TarEntryType.HardLink:
                        return 'l';
                    case TarEntryType.SymbolicLink:
                        return 's';
                    case TarEntryType.CharSpecial:
                        return 'c';
                    case TarEntryType.BlockSpecial:
                        return 'b';
                    case TarEntryType.Directory:
                        return 'd';
                    case TarEntryType.Fifo:
                        return 'p';
                    case TarEntryType.GnuLongLink:
                    case TarEntryType.GnuLongName:
                    case TarEntryType.GnuSparseFile:
                    case TarEntryType.GnuVolumeHeader:
                        return (char)Type;
                    default: return '?';
                }
            }
        }
    }
}
