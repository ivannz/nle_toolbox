import os
import tempfile
import torch

from torch import Tensor, Size
from torch.types import Storage, Device
from typing import Union, BinaryIO


def mkstemp(suffix: str = None, prefix: str = None, dir: str = None) -> str:
    """Create and return a unique temporary file. The caller is responsible
    for deleting the file when done with it.
    """

    fid, tempname = tempfile.mkstemp(suffix, prefix, dir, text=False)
    os.close(fid)

    return tempname


def write_file(
    tensor: Tensor,
    file: BinaryIO,
    *,
    save_size: bool = False,
    is_real_file: bool = None,
) -> None:
    """Write tensor storage to a file-like object.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor, the storage of which to write to a file.

    file : file-like, file descriptor
        The file-like object to write to.

    save_size : bool, default=False
        Whether to write the size of the storage before the binary data.

    Details
    -------
    Reverse engineered logic from
        [writeFile()](/torch/csrc/generic/StorageMethods.cpp#L327-L351)
    which uses
        [writeFileRaw](/torch/csrc/generic/serialization.cpp#L15-L87)

    See Also
    --------
    torch/csrc/generic/StorageMethods.cpp#L462-L487
        * .from_file(filename, shared: bool = False, (size|nbytes): int = 0)
        * .from_buffer(
            buf, byte_order: str = ('native', 'little', 'big'),
            count: int = -1, offset:int = 0, dtype: torch.dtype != None)
        * ._write_file(
            file, is_real_file: bool, save_size: bool, element_size: int)

    """
    # in case we want to override this rather technical flag
    if not isinstance(is_real_file, bool):
        try:
            is_real_file = file.fileno() >= 0

        except (OSError, AttributeError):
            is_real_file = isinstance(file, int)

    # Although, since PR#62030 (oct 2021) this function is implemented through
    #  `UntypedStorage`, which requires the element size positional. The method
    #  `.storage()` returns `TypedStorage`, which knows the `.dtype` and
    #  correctly computes the size.
    tensor.storage()._write_file(file, is_real_file, save_size)


def write(tensor: Tensor, filename: str, *, at: int = None) -> None:
    """Write tensor into a flat binary file by appending or overwriting.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor, the storage of which to write to a file.

    filename : str
        The binary file, which is mapped into the new tensor's storage.

    st : int = None
        The byte position in the file at which to put the tensor's data in
        low-level binary representation. Appends to the end if `at=None`.
    """
    assert at is None or at >= 0

    # 'r+b' opens the file for binary read/write with no truncation
    with open(filename, "r+b") as f:
        # append or write at the specified position
        f.seek(0, 2) if at is None else f.seek(at, 0 if at >= 0 else 2)
        write_file(tensor, f, save_size=False)


def from_file(
    filename: str,
    dtype: torch.dtype,
    shape: Union[Size, list[int], tuple[int]],
    *,
    writeable: bool = False,
    device: Device = None,
) -> Tensor:
    """Create a tensor with data linked to the file.

    Arguments
    ---------
    filename : str
        The binary file, which is mapped into the new tensor's storage.

    dtype : torch.dtype
        The data type of to interprete the file's contents as.

    shape : torch.Size, tuple, or int
        A tuple or `torch.Size` determining the shape of the data in the mapped
        file, or the flat size of the storage in elements of type `dtype`.

    writeable : bool, default=False
        Whether the changes to the storage affect the mapped file.

    device : torch.device, optional)
        The desired device of returned tensor. If None`, uses the current
        device for the default tensor type, e.g. CPU.

    Returns
    -------
    tensor : torch.Tensor
        A new tensor of specified `dtype` and `shape` with memory-mapped
        file as the underlying read-only or writeable storage.

    Details
    -------
    As of pytorch 1.11 memory mapped storage is supported for CPU only.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if isinstance(shape, (tuple, list)):
        shape = Size(shape)

    assert isinstance(shape, Size)

    # create an empty tensor of the proper `dtype` and reassign
    #  its underlying storage to a mem-mapped file-based storage
    # XXX since PR#62030 storage is bytes only and does not track
    #  dtype, delegating this task to tensors. Everything works thru
    # `UntypedStorage`
    tensor = torch.tensor([], dtype=dtype, device=device)
    sto: Storage = tensor.storage()  # actually a `_TypedStorage`
    tensor.set_(sto.from_file(filename, shared=writeable, size=shape.numel()))
    # XXX `_set_from_file` is more hassle since we'll need to call
    #  `torch._utils._element_size(dtype)`. `TypedStorage` knows its `dtype`.

    # reshape unless flat
    if len(shape) > 1:
        return tensor.reshape_(shape)

    return tensor
