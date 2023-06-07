import dataclasses
import sys
import types
from dataclasses import (
    _FIELD,
    _FIELD_CLASSVAR,
    _FIELD_INITVAR,
    MISSING,
    Field,
    _is_classvar,
    _is_initvar,
    _is_type,
    field,
)


def _get_field(cls, a_name, a_type, default_kw_only):
    # Return a Field object for this field name and type.  ClassVars and
    # InitVars are also returned, but marked as such (see f._field_type).
    # default_kw_only is the value of kw_only to use if there isn't a field()
    # that defines it.

    # If the default value isn't derived from Field, then it's only a
    # normal default value.  Convert it to a Field().
    default = getattr(cls, a_name, MISSING)
    if isinstance(default, Field):
        f = default
    else:
        if isinstance(default, types.MemberDescriptorType):
            # This is a field in __slots__, so it has no default value.
            default = MISSING
        f = field(default=default)

    # Only at this point do we know the name and the type.  Set them.
    f.name = a_name
    f.type = a_type

    # Assume it's a normal field until proven otherwise.  We're next
    # going to decide if it's a ClassVar or InitVar, everything else
    # is just a normal field.
    f._field_type = _FIELD

    # In addition to checking for actual types here, also check for
    # string annotations.  get_type_hints() won't always work for us
    # (see https://github.com/python/typing/issues/508 for example),
    # plus it's expensive and would require an eval for every string
    # annotation.  So, make a best effort to see if this is a ClassVar
    # or InitVar using regex's and checking that the thing referenced
    # is actually of the correct type.

    # For the complete discussion, see https://bugs.python.org/issue33453

    # If typing has not been imported, then it's impossible for any
    # annotation to be a ClassVar.  So, only look for ClassVar if
    # typing has been imported by any module (not necessarily cls's
    # module).
    typing = sys.modules.get("typing")
    if typing:
        if _is_classvar(a_type, typing) or (
            isinstance(f.type, str)
            and _is_type(f.type, cls, typing, typing.ClassVar, _is_classvar)
        ):
            f._field_type = _FIELD_CLASSVAR

    # If the type is InitVar, or if it's a matching string annotation,
    # then it's an InitVar.
    if f._field_type is _FIELD:
        # The module we're checking against is the dataclasses module.
        dataclasses = sys.modules["dataclasses"]
        if _is_initvar(a_type, dataclasses) or (
            isinstance(f.type, str)
            and _is_type(f.type, cls, dataclasses, dataclasses.InitVar, _is_initvar)
        ):
            f._field_type = _FIELD_INITVAR

    # Validations for individual fields.  This is delayed until now,
    # instead of in the Field() constructor, since only here do we
    # know the field name, which allows for better error reporting.

    # Special restrictions for ClassVar and InitVar.
    if f._field_type in (_FIELD_CLASSVAR, _FIELD_INITVAR):
        if f.default_factory is not MISSING:
            raise TypeError(f"field {f.name} cannot have a " "default factory")
        # Should I check for other field settings? default_factory
        # seems the most serious to check for.  Maybe add others.  For
        # example, how about init=False (or really,
        # init=<not-the-default-init-value>)?  It makes no sense for
        # ClassVar and InitVar to specify init=<anything>.

    # kw_only validation and assignment.
    if f._field_type in (_FIELD, _FIELD_INITVAR):
        # For real and InitVar fields, if kw_only wasn't specified use the
        # default value.
        if f.kw_only is MISSING:
            f.kw_only = default_kw_only
    else:
        # Make sure kw_only isn't set for ClassVars
        assert f._field_type is _FIELD_CLASSVAR
        if f.kw_only is not MISSING:
            raise TypeError(f"field {f.name} is a ClassVar but specifies " "kw_only")

    return f


def monkey_patch_dataclasses():
    # a hack to repr dataclasses.MISSING as [[MISSING]]
    # instead of <dataclasses._MISSING_TYPE object at 0x7fb503cadb40>
    MISSING_LABEL = "[[MISSING]]"
    type(MISSING).__repr__ = lambda _: MISSING_LABEL  # type: ignore

    # update dataclasses._get_field so that it no longer
    # complains on mutable default values for fields
    dataclasses._get_field = _get_field
