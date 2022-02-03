import re
import numpy
import warnings


class PatternRegistry:
    """A registry of regex _byte_ patterns to be searched in the specified
    fields of the dict observation.
    """
    __patterns__ = None

    def __init__(self, **patterns):
        """Init the registry with the specified patterns, named like

            <source-field>__<name>: regex-pattern

        where the both `source-field` and `name` must be valid Python
        Identifiers.
        """
        self._compiled_patterns = {}

        # get class-level regexes and override them with instance-level ones
        self.register(**(self.__patterns__ or {}))
        self.register(**patterns)

    def register(self, **patterns):
        table = {}
        for pat, rx in patterns.items():
            # get properly named attrs
            source, dunder, ignore = pat.partition('__')
            if not dunder:
                warnings.warn(f"Inappropriately named regex `{pat}`.",
                              RuntimeWarning)
                continue

            # add an extra terminal _under_ to indicate a computed field
            table[pat + '_'] = source, re.compile(rx)

        self._compiled_patterns.update(table)

    def __call__(self, obs):
        """Clean up on `None`, otherwise parse the dict.

        Warning
        -------
        Assumes that the appropriate data is plain `str` or `bytes`.
        """
        if obs is None:
            for pat in self._compiled_patterns:
                if hasattr(self, pat):
                    delattr(self, pat)

        else:
            for pat, (field, rx) in self._compiled_patterns.items():
                setattr(self, pat, re.search(rx, obs[field]))

        return self


class ObservationFlags(PatternRegistry):
    """Apply basic regexes to observation's character data."""
    __patterns__ = {
        'message__ynq': br"\?\s+\[[ynNaq0-9]{2,}\]",
        'tty_chars__more': br"--[Mm]ore--",
    }
