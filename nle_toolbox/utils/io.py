import os
import tempfile


def mkstemp(suffix=None, prefix=None, dir=None):
    """Create and return a unique temporary file. The caller is responsible
    for deleting the file when done with it.
    """

    fid, tempname = tempfile.mkstemp(suffix, prefix, dir, text=False)
    os.close(fid)

    return tempname
