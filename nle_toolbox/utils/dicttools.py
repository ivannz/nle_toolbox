import copy
from collections import defaultdict


def propagate(lookup, tree, *, prefix='', value=None, delim='.'):
    """Assign values to the nodes propagating from parents to children.

    Parameters
    ----------
    lookup : dict
        The lookup table of values to assign to the nodes of the tree.

    tree : iterable
        Linearized structure of the tree: paths from the implicit root to all
        intermediate nodes and leaves. Paths are represented by strings with
        node names separated by `delim` string.

    prefix : str, optional
        The current prefix of the path in the tree.

    value : any, optional
        The value propagated from the parent.

    delim : str, optional
        The delimiter used to separate nodes in the paths.

    Yields
    ------
    path : str
        `delim`-delimited path to the current node.

    value : str
        The value assigned to the current node.

    Details
    -------
    Yields all prefixes of the tree with values taken from `lookup` or
    propagated from the parent.
    """
    # '' is a the parent of all nodes (except itself)
    if '' in tree:
        yield from propagate(lookup, set(n for n in tree if n),
                             prefix=prefix, value=value, delim=delim)
        return

    # lookup (or inherit) the parent's value
    value = lookup.get(prefix, value)  # lookup.get(prefix, 1.) * value
    yield prefix, value

    # collect children of the current prefix (aka `parent`)
    children, prefix = {}, prefix + (delim if prefix else '')
    for node in tree:
        name, dot, child = node.partition(delim)
        children.setdefault(prefix + name, set())
        if child:
            children[prefix + name].add(child)

    # propagate this parent's value to its children
    for prefix, tree in children.items():
        yield from propagate(lookup, tree,
                             prefix=prefix, value=value, delim=delim)


def resolve(references, *, errors='raise'):
    """Resolve references.

    Parameters
    ----------
    references : dict
        The references represented by a dictionary, wherein the value under
        each key indicates the source, either external, or referring back to
        another key.

    errors : str, default='raise'
        Whether to raise a `RecursionError` if a cyclic reference is detected,
        or not. If not and a cycle was detected, then the reference resolution
        result will contain a chain ending in a self-reference.

    Returns
    -------
    resolved : dict
        Resolved references by assigning the root source in the reference tree
        to each key. Contains the same keys as the original `references`.

    Details
    -------
    Simplified DFS for directed tree-like graphs to handle possible cycles.
    Raises `RecursionError` if a cyclic reference is detected.
    """
    assert errors in ('raise', 'ignore')

    # dfs through the table of references
    visited, resolved = set(), {}
    for root in references:
        if root in visited:
            continue

        path = []
        # resolve the keys, that have not been visited yet
        while root not in visited:  # while a key is unseen
            path.append(root)
            visited.add(root)

            root = references[root]
            if root not in references:
                # we reached the end of the reference chain, break out!
                break

        else:  # `root` is visited, but definitely not final
            if root in resolved:
                # fetch the end-point of a resolved reference
                root = resolved[root]

            elif errors != 'ignore':
                # a cyclic reference: visited, and neither final, nor resolved
                raise RecursionError(path)

        resolved.update(dict.fromkeys(path, root))

    return resolved


def override(dictionary, overrides, *, delim='__'):
    """Override the values inside the given nested dictionary.

    Details
    -------
    Creates a shallow copy of the dictionary overrides. Supports
    scikit's `ParameterGrid` syntax for nested overrides.
    """
    if not overrides:
        return copy.deepcopy(dictionary)

    # split overrides into nested and local
    nested, local = defaultdict(dict), {}
    for key, value in overrides.items():
        key, dot, sub_key = key.partition(delim)
        if dot:
            nested[key][sub_key] = value
        else:
            local[key] = value

    # override the non-nested items and possibly introduce new ones
    out = {**dictionary, **local}
    for key, sub_params in nested.items():
        out[key] = override(dictionary.get(key, {}), sub_params, delim=delim)

    return out


def flat_view(value, *prefix, memo=None, depth=None):
    """Depth-first linearize a nested dictionary.

    Arguments
    ---------
    value : any
        The value associated with the prefix or the dictionary, the keys of
        which to linearize in depth-first manner.

    *prefix : hashable
        The variable immutable positionals interpreted as the current key.

    memo : None, or set
        Set of id-s of visited nested dictionaries.

    depth : None, or int
        The maximal allowed linearized key length. `None` means unlimited.

    Yields
    ------
    key : tuple
        The full linearized key. Always a tuple even if the key's value is
        not a nested dictionary in the original dictionary.

    value : any
        The value under the full key in the nested dictionary.
    """
    if memo is None:
        memo = set()

    if isinstance(value, dict) and value and (depth is None or depth > 0):
        if id(value) in memo:
            raise RecursionError(prefix)

        if depth is not None:
            depth = depth - 1

        memo.add(id(value))
        for k, v in value.items():
            yield from flat_view(v, *prefix, k, memo=memo, depth=depth)

        return

    yield prefix, value


def flatten(dictionary, *, depth=None, delim='__'):
    """Depth first redundantly flatten a nested dictionary.

    Arguments
    ---------
    dictionary : dict
        The dictionary to linearize with string keys, that must not contain
        the delimiter.

    depth : None, or int
        The maximal linearization depth. `None` means unlimited.

    delim : str, default='.'
        The delimiter used to indicate nested keys. Defaults to
        torch-compatible module separator '.' (dot).

    Returns
    -------
    flat : dict
        A dictionary with original non-dict values and flattened nested
        dictionaries with keys separated by the delimiter.
    """
    if depth is not None and depth <= 1:
        return dictionary

    out = {}
    for flat, val in flat_view(dictionary, depth=depth):
        # XXX: these checks are sub-optimal
        if not all(isinstance(k, str) for k in flat):
            raise TypeError(f'Non-string key detected `{flat}`.')

        if any(delim in k for k in flat):
            raise ValueError(f'No key must contain `{delim}`. Got `{flat}`.')

        out[delim.join(flat)] = val

    return out


def unflatten(dictionary, *, delim='__'):
    """Breadth first turn flattened dictionary into a nested one.

    Arguments
    ---------
    dictionary : dict
        The dictionary to traverse and linearize.

    delim : str, default='.'
        The delimiter used to indicate nested keys.
    """

    out = defaultdict(dict)

    # try to maintain current order of the dictionary
    for key, value in dictionary.items():
        key, sep, sub_key = key.partition(delim)
        if sep:
            out[key][sub_key] = value
        else:
            out[key] = value

    for key, value in out.items():
        if isinstance(value, dict):
            out[key] = unflatten(value, delim=delim)

    return dict(out)


def collate(records):
    """Turn a list of dicts into a dict of lists.

    Parameters
    ----------
    records : list of dicts
        An iterable of key-value mappings.

    Returns
    -------
    output : dict of lists
        A dict with all keys ever encountered in the records and values
        collated into lists.
    """
    tmp, inx, n_records = {}, {}, 0
    for j, record in enumerate(records):
        for k, v in record.items():
            tmp.setdefault(k, []).append(v)
            inx.setdefault(k, []).append(j)
        n_records += 1

    # no missing records, many empty records, or no records at all
    if all(sz == n_records for sz in map(len, tmp.values())):
        return tmp

    out = {k: [None] * n_records for k in tmp}
    for k, a in out.items():
        for j, v in zip(inx[k], tmp[k]):
            a[j] = v

    return out
