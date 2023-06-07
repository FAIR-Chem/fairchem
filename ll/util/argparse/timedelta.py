import datetime
import re

# https://gist.github.com/jnothman/4057689
class TimeDeltaType:
    """
    Interprets a string as a timedelta for argument parsing.

    With no default unit:
    >>> tdtype = TimeDeltaType()
    >>> tdtype('5s')
    datetime.timedelta(0, 5)
    >>> tdtype('5.5s')
    datetime.timedelta(0, 5, 500000)
    >>> tdtype('5:06:07:08s')
    datetime.timedelta(5, 22028)
    >>> tdtype('5d06h07m08s')
    datetime.timedelta(5, 22028)
    >>> tdtype('5d')
    datetime.timedelta(5)

    With a default unit of minutes:
    >>> tdmins = TimeDeltaType('m')
    >>> tdmins('5s')
    datetime.timedelta(0, 5)
    >>> tdmins('5')
    datetime.timedelta(0, 300)
    >>> tdmins('6:05')
    datetime.timedelta(0, 21900)

    And some error cases:
    >>> tdtype('5')
    Traceback (most recent call last):
        ...
    ValueError: Cannot infer units for '5'
    >>> tdtype('5:5d')
    Traceback (most recent call last):
        ...
    ValueError: Colon not handled for unit 'd'
    >>> tdtype('5:5ms')
    Traceback (most recent call last):
        ...
    ValueError: Colon not handled for unit 'ms'
    >>> tdtype('5q')
    Traceback (most recent call last):
        ...
    ValueError: Unknown unit: 'q'
    """

    units = {
        "d": datetime.timedelta(days=1),
        "h": datetime.timedelta(seconds=60 * 60),
        "m": datetime.timedelta(seconds=60),
        "s": datetime.timedelta(seconds=1),
        "ms": datetime.timedelta(microseconds=1000),
    }
    colon_mult_ind = ["h", "m", "s"]
    colon_mults = [24, 60, 60]
    unit_re = re.compile(r"[^\d:.,-]+", re.UNICODE)

    def __init__(self, default_unit=None):
        self.default_unit = default_unit

    def __call__(self, val):
        res = datetime.timedelta()
        for num, unit in self._parse(val):
            unit = unit.lower()

            if ":" in num:
                try:
                    colon_mults = self.colon_mults[
                        : self.colon_mult_ind.index(unit) + 1
                    ]
                except ValueError:
                    raise ValueError("Colon not handled for unit %r" % unit)
            else:
                colon_mults = []

            try:
                unit = self.units[unit]
            except KeyError:
                raise ValueError("Unknown unit: %r" % unit)

            mult = 1
            for part in reversed(num.split(":")):
                res += self._mult_td(
                    unit, (float(part) if "." in part else int(part)) * mult
                )
                if colon_mults:
                    mult *= colon_mults.pop()
        return res

    def _parse(self, val):
        pairs = []
        start = 0
        for match in self.unit_re.finditer(val):
            num = val[start : match.start()]
            unit = match.group()
            pairs.append((num, unit))
            start = match.end()
        num = val[start:]
        if num:
            if pairs or self.default_unit is None:
                raise ValueError("Cannot infer units for %r" % num)
            else:
                pairs.append((num, self.default_unit))
        return pairs

    @staticmethod
    def _mult_td(td, mult):
        # Necessary because timedelta * float is not supported:
        return datetime.timedelta(
            days=td.days * mult,
            seconds=td.seconds * mult,
            microseconds=td.microseconds * mult,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
