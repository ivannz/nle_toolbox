import os
import tempfile


def mkstemp(suffix: str = None, prefix: str = None, dir: str = None) -> str:
    """Create and return a unique temporary file. The caller is responsible
    for deleting the file when done with it.
    """

    fid, tempname = tempfile.mkstemp(suffix, prefix, dir, text=False)
    os.close(fid)

    return tempname
