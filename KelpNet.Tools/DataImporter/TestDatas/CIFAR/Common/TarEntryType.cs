namespace KelpNet.Tools
{
    public enum TarEntryType : byte
    {
        File_Old = 0,
        File = 48,
        HardLink = 49,
        SymbolicLink = 50,
        CharSpecial = 51,
        BlockSpecial = 52,
        Directory = 53,
        Fifo = 54,
        File_Contiguous = 55,
        GnuLongLink = (byte)'K',    // "././@LongLink"
        GnuLongName = (byte)'L',    // "././@LongLink"
        GnuSparseFile = (byte)'S',
        GnuVolumeHeader = (byte)'V'
    }
}
