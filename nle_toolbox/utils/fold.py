from numpy.lib.stride_tricks import as_strided as npy_as_strided


def npy_fold2d(
    array,
    /,
    k=1,
    *,
    n_leading=1,
    writeable=True,
):
    """Zero-copy sliding window view for numpy arrays.
    """

    # XXX shouldn't we call it `n_leading`?
    n_leading = (n_leading + array.ndim) if n_leading < 0 else n_leading

    if array.ndim < n_leading + 2:
        raise ValueError(f"Not enough dimensions for 2d folding `{array.shape}`.")

    d0, d1, *shape = array.shape[n_leading:]
    s0, s1, *strides = array.strides[n_leading:]
    return npy_as_strided(
        array, (
            *array.shape[:n_leading],
            d0 - 2 * k, d1 - 2 * k,  # n' = n - w + 1, w = k + 1 + k
            k + 1 + k, k + 1 + k,
            *shape,
        ), (
            *array.strides[:n_leading],
            s0, s1,
            s0, s1,
            *strides,
        ), writeable=writeable,
    )


def pyt_fold2d(
    tensor,
    /,
    k=1,
    *,
    n_leading=1,
    writeable=None,
):
    """Zero-copy sliding window view for torch tensors.
    """
    if writeable is not None:
        raise TypeError("torch does not support access flags in `.as_strided`.")

    # XXX shouldn't we call it `n_leading`?
    n_leading = (n_leading + tensor.ndim) if n_leading < 0 else n_leading

    if tensor.ndim < n_leading + 2:
        raise ValueError(f"Not enough dimensions for 2d folding `{tensor.shape}`.")

    d0, d1, *shape = tensor.shape[n_leading:]
    s0, s1, *strides = tensor.stride()[n_leading:]
    return tensor.as_strided(
        (
            *tensor.shape[:n_leading],
            d0 - 2 * k, d1 - 2 * k,  # n' = n - w + 1, w = k + 1 + k
            k + 1 + k, k + 1 + k,
            *shape,
        ), (
            *tensor.stride()[:n_leading],
            s0, s1,
            s0, s1,
            *strides,
        ),
    )
