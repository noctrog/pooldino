"""Declarative experiment naming library.

Define experiment-name schemas with dataclass-style field helpers::

    from typing import Literal

    class MySpec(ExperimentSpec, examples=("fast-32-kl1e-3-hl",)):
        variant: Literal["fast", "med"] = positional()
        toks: int = positional()
        kl: float | None = option("kl")
        hl: bool = flag()

    spec = MySpec.parse("fast-32-kl1e-3-hl")
    spec.to_name()  # "fast-32-hl-kl0.001"

Design constraints
------------------
This library intentionally uses a small, readable grammar rather than a fully
escaped serializer. Field values must serialize to safe experiment-name atoms.
Arbitrary strings with spaces, slashes, commas, or normal hyphens are rejected.
Float exponent hyphens such as ``1e-05`` are supported.
"""

from __future__ import annotations

import dataclasses
import math
import re
import types
from dataclasses import dataclass, fields
from difflib import get_close_matches
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

try:  # Python 3.11+
    from typing import Self, dataclass_transform
except ImportError:  # pragma: no cover
    from typing_extensions import Self, dataclass_transform

__all__ = [
    "Codec",
    "ExperimentParseError",
    "ExperimentSpec",
    "flag",
    "option",
    "positional",
    "register_codec",
]

T = TypeVar("T")

_MISSING = object()
_EXPERIMENT_METADATA_KEY = "pooldino.experiment"

_TAG_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_ATOM_CHARS_RE = re.compile(r"[A-Za-z0-9_.+\-]+\Z")
_TEXT_ATOM_RE = re.compile(r"[A-Za-z0-9_.+]+\Z")
_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_FLOAT_RE = re.compile(
    r"(?:"
    r"(?:[0-9]+(?:\.[0-9]*)?)"
    r"|"
    r"(?:\.[0-9]+)"
    r")"
    r"(?:[eE][+-]?[0-9]+)?\Z"
)

_RESERVED_FIELD_NAMES = frozenset(
    {
        "_schema",
        "from_name",
        "help",
        "help_text",
        "name",
        "parse",
        "parse_or_raise",
        "register_codec",
        "to_name",
        "try_parse",
        "validate",
    }
)


# ---------------------------------------------------------------------------
# Token grammar
# ---------------------------------------------------------------------------


def _split_name(name: str) -> list[str]:
    """Split an experiment name while preserving float exponents like ``1e-05``."""
    if name == "":
        return []

    tokens: list[str] = []
    start = 0
    for idx, char in enumerate(name):
        if char != "-":
            continue
        is_negative_exponent = (
            idx > 1
            and name[idx - 1] in {"e", "E"}
            and (name[idx - 2].isdigit() or name[idx - 2] == ".")
            and idx + 1 < len(name)
            and name[idx + 1].isdigit()
        )
        if is_negative_exponent:
            continue
        tokens.append(name[start:idx])
        start = idx + 1
    tokens.append(name[start:])
    return tokens


def _is_safe_atom(text: str) -> bool:
    """Return True if text is one legal experiment-name atom."""
    return bool(text and _ATOM_CHARS_RE.fullmatch(text) and _split_name(text) == [text])


def _validate_tag(owner: str, tag: str) -> None:
    if not isinstance(tag, str):
        raise TypeError(f"{owner} tag must be str; got {type(tag).__name__}")
    if not _TAG_RE.fullmatch(tag):
        raise TypeError(f"{owner} tag must match {_TAG_RE.pattern!r}; got {tag!r}")


def _validate_prefix(owner: str, prefix: str | None) -> None:
    if prefix is None:
        return
    if not isinstance(prefix, str):
        raise TypeError(f"{owner} prefix must be str; got {type(prefix).__name__}")
    if not _TAG_RE.fullmatch(prefix):
        raise TypeError(
            f"{owner} prefix must match {_TAG_RE.pattern!r}; got {prefix!r}"
        )


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Codec(Generic[T]):
    """Parser/renderer for one scalar experiment-name value type."""

    name: str
    python_type: type[T]
    parse: Callable[[str], T]
    serialize: Callable[[Any], str] = str
    validate: Callable[[T], None] | None = None


def _parse_str(value: str) -> str:
    if not _TEXT_ATOM_RE.fullmatch(value):
        raise ValueError(value)
    return value


def _parse_int(value: str) -> int:
    if not _INT_RE.fullmatch(value):
        raise ValueError(value)
    return int(value)


def _validate_float(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("float experiment values must be finite")
    if value < 0 or math.copysign(1.0, value) < 0:
        raise ValueError("float experiment values must be non-negative")


def _parse_float(value: str) -> float:
    if not _FLOAT_RE.fullmatch(value):
        raise ValueError(value)
    parsed = float(value)
    _validate_float(parsed)
    return parsed


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(value)


def _serialize_float(value: Any) -> str:
    value = float(value)
    _validate_float(value)
    return repr(value)


def _serialize_bool(value: bool) -> str:
    return "true" if value else "false"


_CODECS: dict[type[Any], Codec[Any]] = {
    int: Codec(name="int", python_type=int, parse=_parse_int),
    float: Codec(
        name="float",
        python_type=float,
        parse=_parse_float,
        serialize=_serialize_float,
        validate=_validate_float,
    ),
    str: Codec(name="str", python_type=str, parse=_parse_str),
    bool: Codec(name="bool", python_type=bool, parse=_parse_bool, serialize=_serialize_bool),
}


def register_codec(python_type: type[T], codec: Codec[T]) -> None:
    """Register or replace the default codec for ``python_type``."""
    if codec.python_type is not python_type:
        raise TypeError(
            f"codec.python_type must be {python_type.__name__}; "
            f"got {codec.python_type.__name__}"
        )
    _CODECS[python_type] = codec  # type: ignore[assignment]


def _run_codec_validate(codec: Codec[Any], value: Any) -> None:
    if codec.validate is not None:
        codec.validate(value)


def _is_valid_type(value: Any, codec: Codec[Any]) -> bool:
    if codec.python_type is float:
        return isinstance(value, (float, int)) and not isinstance(value, bool)
    if codec.python_type is int:
        return type(value) is int
    if codec.python_type is bool:
        return type(value) is bool
    if codec.python_type is str:
        return type(value) is str
    return isinstance(value, codec.python_type)


def _serialize_checked(*, owner: str, value: Any, codec: Codec[Any]) -> str:
    rendered = codec.serialize(value)
    if not isinstance(rendered, str):
        raise ValueError(
            f"{owner} codec {codec.name!r} must serialize to str; "
            f"got {type(rendered).__name__}"
        )
    if not _is_safe_atom(rendered):
        raise ValueError(
            f"{owner} serializes to invalid experiment atom {rendered!r}; "
            "values must not contain token separators or unsafe characters"
        )
    return rendered


def _validate_value(
    *,
    owner: str,
    value: Any,
    codec: Codec[Any],
    choices: tuple[Any, ...] | None,
) -> None:
    if not _is_valid_type(value, codec):
        raise ValueError(
            f"{owner} must be a {codec.name}; got {type(value).__name__}"
        )

    try:
        _run_codec_validate(codec, value)
    except ValueError as exc:
        raise ValueError(f"{owner} is invalid: {exc}") from None

    if choices is not None and value not in choices:
        raise ValueError(
            f"{owner} must be one of {_choices_display(choices)}; got {value!r}"
            f"{_suggestion(str(value), [str(choice) for choice in choices])}"
        )

    rendered = _serialize_checked(owner=owner, value=value, codec=codec)
    try:
        reparsed = codec.parse(rendered)
        _run_codec_validate(codec, reparsed)
    except ValueError as exc:
        raise ValueError(
            f"{owner} serializes to {rendered!r}, which the codec cannot parse: {exc}"
        ) from None

    if reparsed != value:
        raise ValueError(
            f"{owner} does not round-trip through codec {codec.name!r}: "
            f"{value!r} -> {rendered!r} -> {reparsed!r}"
        )


def _try_parse_codec(codec: Codec[Any], value: str) -> Any:
    parsed = codec.parse(value)
    if not _is_valid_type(parsed, codec):
        raise ValueError(
            f"codec {codec.name!r} parsed {value!r} into "
            f"{type(parsed).__name__}, expected {codec.python_type.__name__}"
        )
    _run_codec_validate(codec, parsed)
    return parsed


def _codec_can_parse(codec: Codec[Any], value: str) -> bool:
    try:
        _try_parse_codec(codec, value)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Field descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PositionalDescriptor:
    choices: tuple[Any, ...] | None = None
    group_next: bool = False
    prefix: str | None = None
    codec: Codec[Any] | None = None


@dataclass(frozen=True)
class _OptionDescriptor:
    tag: str | None = None
    choices: tuple[Any, ...] | None = None
    bare_value: Any = _MISSING
    emit_default: bool = False
    codec: Codec[Any] | None = None


@dataclass(frozen=True)
class _FlagDescriptor:
    tag: str | None = None


def _normalize_choices(choices: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if choices is None:
        return None
    if isinstance(choices, (str, bytes)):
        raise TypeError("choices must be a sequence of values, not a string/bytes object")
    normalized = tuple(choices)
    if not normalized:
        raise TypeError("choices must not be empty")
    return normalized


def positional(
    *,
    choices: Sequence[Any] | None = None,
    group_next: bool = False,
    prefix: str | None = None,
    codec: Codec[Any] | None = None,
) -> Any:
    """Declare a required positional experiment-name field."""
    return dataclasses.field(
        metadata={
            _EXPERIMENT_METADATA_KEY: _PositionalDescriptor(
                choices=_normalize_choices(choices),
                group_next=group_next,
                prefix=prefix,
                codec=codec,
            )
        }
    )


def option(
    tag: str | None = None,
    *,
    default: Any = None,
    bare: Any = _MISSING,
    emit_default: bool = False,
    choices: Sequence[Any] | None = None,
    codec: Codec[Any] | None = None,
) -> Any:
    """Declare an optional tagged field."""
    return dataclasses.field(
        default=default,
        metadata={
            _EXPERIMENT_METADATA_KEY: _OptionDescriptor(
                tag=tag,
                choices=_normalize_choices(choices),
                bare_value=bare,
                emit_default=emit_default,
                codec=codec,
            )
        },
    )


def flag(tag: str | None = None) -> Any:
    """Declare a boolean presence flag."""
    return dataclasses.field(
        default=False,
        metadata={_EXPERIMENT_METADATA_KEY: _FlagDescriptor(tag=tag)},
    )


# ---------------------------------------------------------------------------
# Resolved schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PositionalField:
    name: str
    codec: Codec[Any]
    choices: tuple[Any, ...] | None
    group_next: bool
    prefix: str | None


@dataclass(frozen=True)
class _OptionField:
    name: str
    codec: Codec[Any]
    tag: str
    choices: tuple[Any, ...] | None
    bare_value: Any = _MISSING
    absent_default: Any = _MISSING
    emit_default: bool = False


@dataclass(frozen=True)
class _FlagField:
    name: str
    tag: str


@dataclass(frozen=True)
class _SinglePositional:
    field: _PositionalField


@dataclass(frozen=True)
class _GroupedPositionals:
    first: _PositionalField
    second: _PositionalField
    values_by_token: Mapping[str, tuple[Any, Any]]


_PositionalUnit = _SinglePositional | _GroupedPositionals


@dataclass(frozen=True)
class _Schema:
    positionals: tuple[_PositionalField, ...]
    positional_units: tuple[_PositionalUnit, ...]
    options: tuple[_OptionField, ...]
    flags: tuple[_FlagField, ...]
    examples: tuple[str, ...]
    flags_by_tag: Mapping[str, _FlagField]
    options_by_tag_length_desc: tuple[_OptionField, ...]
    known_tags: tuple[str, ...]


def _strip_annotated(type_hint: Any) -> Any:
    while get_origin(type_hint) is Annotated:
        type_hint = get_args(type_hint)[0]
    return type_hint


def _strip_none_union(type_hint: Any) -> Any:
    """Return the non-None member of ``T | None`` when there is exactly one."""
    type_hint = _strip_annotated(type_hint)
    origin = get_origin(type_hint)
    if origin in {Union, types.UnionType}:
        args = get_args(type_hint)
        non_none = [arg for arg in args if _strip_annotated(arg) is not type(None)]
        if len(non_none) == 1:
            return _strip_annotated(non_none[0])
    return type_hint


def _type_allows_none(type_hint: Any) -> bool:
    type_hint = _strip_annotated(type_hint)
    origin = get_origin(type_hint)
    if origin not in {Union, types.UnionType}:
        return False
    return any(_strip_annotated(arg) is type(None) for arg in get_args(type_hint))


def _literal_values(type_hint: Any) -> tuple[Any, ...] | None:
    type_hint = _strip_none_union(type_hint)
    if get_origin(type_hint) is not Literal:
        return None
    return tuple(get_args(type_hint))


def _resolve_codec(
    type_hint: Any,
    *,
    context: str,
    explicit_codec: Codec[Any] | None = None,
) -> Codec[Any]:
    if explicit_codec is not None:
        return explicit_codec

    type_hint = _strip_none_union(type_hint)
    if get_origin(type_hint) is Literal:
        values = get_args(type_hint)
        if not values:
            raise TypeError(f"Experiment {context} has an empty Literal")
        literal_types = {type(value) for value in values}
        if len(literal_types) != 1:
            type_names = ", ".join(sorted(t.__name__ for t in literal_types))
            raise TypeError(
                f"Experiment {context} Literal choices must all have one type; "
                f"got {type_names}"
            )
        type_hint = next(iter(literal_types))

    if type_hint in _CODECS:
        return _CODECS[type_hint]
    raise TypeError(f"Experiment {context} has unsupported type {type_hint!r}")


def _field_tag(name: str, tag: str | None) -> str:
    if tag is not None:
        return tag
    return name[:-1] if name.endswith("_") else name


def _choices_display(choices: tuple[Any, ...] | None) -> str:
    if not choices:
        return ""
    return ", ".join(str(choice) for choice in choices)


def _choices_text(choices: tuple[Any, ...] | None) -> str:
    if not choices:
        return ""
    return "choices: " + _choices_display(choices)


def _suggestion(value: str, choices: Sequence[str] | None) -> str:
    if not choices:
        return ""
    matches = get_close_matches(value, list(choices), n=1, cutoff=0.5)
    if not matches:
        return ""
    return f"; did you mean {matches[0]!r}?"


def _tag_suggestion(token: str, choices: Sequence[str]) -> str:
    suggestion = _suggestion(token, choices)
    if suggestion:
        return suggestion
    tag_prefix = token.rstrip("0123456789.+-")
    if tag_prefix and tag_prefix != token:
        return _suggestion(tag_prefix, choices)
    return ""


def _validate_choices(
    *,
    owner: str,
    choices: tuple[Any, ...] | None,
    codec: Codec[Any],
) -> None:
    if choices is None:
        return

    rendered_seen: dict[str, Any] = {}
    for choice in choices:
        try:
            _validate_value(
                owner=f"{owner} choice {choice!r}",
                value=choice,
                codec=codec,
                choices=None,
            )
        except ValueError as exc:
            raise TypeError(str(exc)) from None

        rendered = _serialize_checked(
            owner=f"{owner} choice {choice!r}",
            value=choice,
            codec=codec,
        )
        if rendered in rendered_seen:
            raise TypeError(
                f"{owner} has duplicate serialized choice {rendered!r} "
                f"for {rendered_seen[rendered]!r} and {choice!r}"
            )
        rendered_seen[rendered] = choice


def _merge_choices(
    explicit: tuple[Any, ...] | None,
    literal: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    if explicit is None:
        return literal
    if literal is not None and explicit != literal:
        raise TypeError(
            f"explicit choices {explicit!r} conflict with Literal choices {literal!r}"
        )
    return explicit


def _raw_annotations(cls: type[Any]) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        annotations.update(getattr(base, "__annotations__", {}))
    return annotations


def _get_type_hints_lenient(cls: type[Any]) -> dict[str, Any]:
    try:
        return get_type_hints(cls, include_extras=True)
    except (NameError, AttributeError, TypeError):
        return _raw_annotations(cls)


def _build_positional_units(
    cls: type[Any],
    positionals: tuple[_PositionalField, ...],
) -> tuple[_PositionalUnit, ...]:
    units: list[_PositionalUnit] = []
    i = 0
    while i < len(positionals):
        pf = positionals[i]
        if not pf.group_next:
            units.append(_SinglePositional(pf))
            i += 1
            continue

        if i + 1 >= len(positionals):
            raise TypeError(
                f"{cls.__name__}.{pf.name} uses group_next=True but has no "
                "following positional field"
            )
        next_pf = positionals[i + 1]
        if next_pf.group_next:
            raise TypeError(
                f"{cls.__name__}.{next_pf.name} cannot use group_next=True because "
                f"it is already grouped by {cls.__name__}.{pf.name}"
            )
        if pf.codec.python_type is not str or next_pf.codec.python_type is not str:
            raise TypeError(
                f"{cls.__name__}.{pf.name} uses group_next=True but grouped "
                "positionals must both be strings"
            )
        if pf.prefix is not None or next_pf.prefix is not None:
            raise TypeError(
                f"{cls.__name__}.{pf.name} uses group_next=True but grouped "
                "positionals cannot have prefixes"
            )
        if not pf.choices or not next_pf.choices:
            raise TypeError(
                f"{cls.__name__}.{pf.name} uses group_next=True but both "
                "grouped fields must define finite choices"
            )

        values_by_token: dict[str, tuple[Any, Any]] = {}
        for first in pf.choices:
            for second in next_pf.choices:
                token = f"{pf.codec.serialize(first)}{next_pf.codec.serialize(second)}"
                if token in values_by_token:
                    previous = values_by_token[token]
                    raise TypeError(
                        f"{cls.__name__}.{pf.name}+{next_pf.name} has ambiguous "
                        f"grouped token {token!r}: {previous!r} and {(first, second)!r}"
                    )
                values_by_token[token] = (first, second)

        units.append(
            _GroupedPositionals(
                first=pf,
                second=next_pf,
                values_by_token=types.MappingProxyType(values_by_token),
            )
        )
        i += 2

    return tuple(units)


def _value_exactly_parses(opt: _OptionField, raw: str) -> bool:
    if raw == "":
        return opt.bare_value is not _MISSING
    try:
        parsed = _try_parse_codec(opt.codec, raw)
    except ValueError:
        return False
    return opt.choices is None or parsed in opt.choices


def _value_prefix_might_parse(opt: _OptionField, prefix: str) -> bool:
    if prefix == "":
        return opt.bare_value is not _MISSING

    if opt.choices is not None:
        rendered_choices = [
            _serialize_checked(
                owner=f"option '-{opt.tag}' choice",
                value=choice,
                codec=opt.codec,
            )
            for choice in opt.choices
        ]
        return any(choice.startswith(prefix) for choice in rendered_choices)

    if opt.codec.python_type is str:
        return bool(_TEXT_ATOM_RE.fullmatch(prefix))
    if opt.codec.python_type is bool:
        return "true".startswith(prefix) or "false".startswith(prefix)
    if opt.codec.python_type is int:
        return bool(re.fullmatch(r"[0-9]+", prefix))
    if opt.codec.python_type is float:
        probes = ("", "0", "1", ".0", "e0", "e-0", "e+0")
        return any(_codec_can_parse(opt.codec, prefix + probe) for probe in probes)

    if _is_safe_atom(prefix):
        return True
    return _codec_can_parse(opt.codec, prefix)


def _validate_schema(cls: type[Any], schema: _Schema) -> None:
    tag_owners: dict[str, str] = {}

    for fl in schema.flags:
        _validate_tag(f"{cls.__name__}.{fl.name}", fl.tag)
        owner = f"flag '{fl.name}'"
        if fl.tag in tag_owners:
            raise TypeError(
                f"{cls.__name__} has duplicate experiment tag {fl.tag!r} "
                f"for {tag_owners[fl.tag]} and {owner}"
            )
        tag_owners[fl.tag] = owner

    for opt in schema.options:
        _validate_tag(f"{cls.__name__}.{opt.name}", opt.tag)
        owner = f"option '{opt.name}'"
        if opt.tag in tag_owners:
            raise TypeError(
                f"{cls.__name__} has duplicate experiment tag {opt.tag!r} "
                f"for {tag_owners[opt.tag]} and {owner}"
            )
        tag_owners[opt.tag] = owner

    for pf in schema.positionals:
        _validate_prefix(f"{cls.__name__}.{pf.name}", pf.prefix)

    for opt in schema.options:
        _validate_choices(
            owner=f"{cls.__name__}.{opt.name}",
            choices=opt.choices,
            codec=opt.codec,
        )

        if opt.absent_default is not _MISSING and opt.absent_default is not None:
            try:
                _validate_value(
                    owner=f"{cls.__name__}.{opt.name} default",
                    value=opt.absent_default,
                    codec=opt.codec,
                    choices=opt.choices,
                )
            except ValueError as exc:
                raise TypeError(str(exc)) from None

        if opt.bare_value is not _MISSING:
            if opt.bare_value is None:
                raise TypeError(
                    f"{cls.__name__}.{opt.name} bare value cannot be None; "
                    "a bare tag should encode a value distinct from absence"
                )
            try:
                _validate_value(
                    owner=f"{cls.__name__}.{opt.name} bare value",
                    value=opt.bare_value,
                    codec=opt.codec,
                    choices=opt.choices,
                )
            except ValueError as exc:
                raise TypeError(str(exc)) from None

    for pf in schema.positionals:
        _validate_choices(
            owner=f"{cls.__name__}.{pf.name}",
            choices=pf.choices,
            codec=pf.codec,
        )

    sorted_options = sorted(schema.options, key=lambda opt: opt.tag)
    for i, first in enumerate(sorted_options):
        for second in sorted_options[i + 1 :]:
            shorter, longer = (first, second)
            if len(second.tag) < len(first.tag):
                shorter, longer = second, first
            if not longer.tag.startswith(shorter.tag):
                continue
            suffix = longer.tag[len(shorter.tag) :]
            if _value_prefix_might_parse(shorter, suffix):
                raise TypeError(
                    f"{cls.__name__} has ambiguous option tags "
                    f"{shorter.tag!r} and {longer.tag!r}"
                )

    for opt in schema.options:
        for fl in schema.flags:
            if not fl.tag.startswith(opt.tag):
                continue
            suffix = fl.tag[len(opt.tag) :]
            if _value_exactly_parses(opt, suffix):
                raise TypeError(
                    f"{cls.__name__} has ambiguous tag {fl.tag!r}: it can be "
                    f"flag '{fl.name}' or option '{opt.name}' with value {suffix!r}"
                )


def _build_schema(cls: type[Any], examples: tuple[str, ...]) -> _Schema:
    """Introspect experiment fields to build and validate a parsing schema."""
    hints = _get_type_hints_lenient(cls)
    positionals: list[_PositionalField] = []
    options: list[_OptionField] = []
    flags: list[_FlagField] = []

    for f in fields(cls):  # type: ignore[arg-type]
        descriptor = f.metadata.get(_EXPERIMENT_METADATA_KEY)
        if descriptor is None:
            if f.init or f.compare:
                raise TypeError(
                    f"{cls.__name__}.{f.name} is a dataclass field but is not "
                    "declared with positional(), option(), or flag(). Use ClassVar "
                    "or dataclasses.field(init=False, compare=False, ...) for "
                    "non-name state."
                )
            continue

        if f.name in _RESERVED_FIELD_NAMES:
            raise TypeError(
                f"{cls.__name__}.{f.name} is reserved by ExperimentSpec; "
                "choose a different field name"
            )
        if f.name not in hints:
            raise TypeError(f"{cls.__name__}.{f.name} is missing a type annotation")

        field_type = hints[f.name]
        literal_choices = _literal_values(field_type)

        if isinstance(descriptor, _PositionalDescriptor):
            codec = _resolve_codec(
                field_type,
                context=f"positional '{f.name}'",
                explicit_codec=descriptor.codec,
            )
            positionals.append(
                _PositionalField(
                    name=f.name,
                    codec=codec,
                    choices=_merge_choices(descriptor.choices, literal_choices),
                    group_next=descriptor.group_next,
                    prefix=descriptor.prefix,
                )
            )
            continue

        if isinstance(descriptor, _FlagDescriptor):
            codec = _resolve_codec(field_type, context=f"flag '{f.name}'")
            if codec.python_type is not bool or _type_allows_none(field_type):
                raise TypeError(f"Experiment flag '{f.name}' must be annotated as bool")
            flags.append(_FlagField(name=f.name, tag=_field_tag(f.name, descriptor.tag)))
            continue

        if isinstance(descriptor, _OptionDescriptor):
            if f.default is None and not _type_allows_none(field_type):
                raise TypeError(
                    f"{cls.__name__}.{f.name} uses option() with default None but "
                    f"is annotated as {field_type!r}; use T | None or provide a "
                    "non-None default"
                )
            codec = _resolve_codec(
                field_type,
                context=f"option '{f.name}'",
                explicit_codec=descriptor.codec,
            )
            options.append(
                _OptionField(
                    name=f.name,
                    codec=codec,
                    tag=_field_tag(f.name, descriptor.tag),
                    choices=_merge_choices(descriptor.choices, literal_choices),
                    bare_value=descriptor.bare_value,
                    absent_default=(
                        f.default if f.default is not dataclasses.MISSING else _MISSING
                    ),
                    emit_default=descriptor.emit_default,
                )
            )
            continue

        raise TypeError(f"{cls.__name__}.{f.name} has invalid experiment metadata")

    positionals_tuple = tuple(positionals)
    schema = _Schema(
        positionals=positionals_tuple,
        positional_units=_build_positional_units(cls, positionals_tuple),
        options=tuple(options),
        flags=tuple(flags),
        examples=examples,
        flags_by_tag=types.MappingProxyType({fl.tag: fl for fl in flags}),
        options_by_tag_length_desc=tuple(
            sorted(options, key=lambda opt: len(opt.tag), reverse=True)
        ),
        known_tags=tuple(sorted([*(fl.tag for fl in flags), *(opt.tag for opt in options)])),
    )
    _validate_schema(cls, schema)
    return schema


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class _PositionalParse:
    values: dict[str, Any]
    remaining: list[str]


@dataclass(frozen=True)
class _TokenParse:
    name: str
    value: Any


class ExperimentParseError(ValueError):
    """Raised when an experiment name is invalid for a spec."""

    def __init__(
        self,
        *,
        spec_name: str,
        name: str,
        reason: str,
        help_text: str,
    ) -> None:
        self.spec_name = spec_name
        self.name = name
        self.reason = reason
        self.help_text = help_text
        super().__init__(
            f"Could not parse {name!r} as {spec_name}: {reason}\n\n{help_text}"
        )


def _expected_value(pf: _PositionalField) -> str:
    if pf.choices:
        return "one of " + _choices_display(pf.choices)
    if pf.prefix is not None:
        return f"{pf.prefix}<{pf.codec.name}>"
    return pf.codec.name


def _parse_scalar_value(
    *,
    raw: str,
    owner: str,
    codec: Codec[Any],
    choices: tuple[Any, ...] | None,
) -> Any:
    try:
        parsed = _try_parse_codec(codec, raw)
    except ValueError:
        raise _ParseError(f"invalid {codec.name} value {raw!r} for {owner}") from None

    if choices is not None and parsed not in choices:
        raise _ParseError(
            f"invalid value {parsed!r} for {owner}; expected one of "
            f"{_choices_display(choices)}"
            f"{_suggestion(str(parsed), [str(choice) for choice in choices])}"
        )

    try:
        _validate_value(owner=owner, value=parsed, codec=codec, choices=choices)
    except ValueError as exc:
        raise _ParseError(str(exc)) from None
    return parsed


def _parse_single_positional(token: str, pf: _PositionalField) -> Any:
    raw = token
    if pf.prefix is not None:
        if not token.startswith(pf.prefix):
            raise _ParseError(
                f"invalid token {token!r} for field '{pf.name}'; "
                f"expected prefix {pf.prefix!r}"
            )
        raw = token[len(pf.prefix) :]
        if raw == "":
            raise _ParseError(
                f"invalid token {token!r} for field '{pf.name}'; "
                f"expected value after prefix {pf.prefix!r}"
            )

    return _parse_scalar_value(
        raw=raw,
        owner=f"field '{pf.name}'",
        codec=pf.codec,
        choices=pf.choices,
    )


def _parse_grouped_positionals(
    token: str,
    unit: _GroupedPositionals,
) -> tuple[Any, Any]:
    pair = unit.values_by_token.get(token)
    if pair is not None:
        return pair

    first = _choices_text(unit.first.choices) or unit.first.codec.name
    second = _choices_text(unit.second.choices) or unit.second.codec.name
    raise _ParseError(
        f"invalid grouped token {token!r} for fields "
        f"'{unit.first.name}' and '{unit.second.name}' ({first}; {second})"
        f"{_suggestion(token, list(unit.values_by_token))}"
    )


def _parse_positionals(
    tokens: list[str],
    positional_units: tuple[_PositionalUnit, ...],
) -> _PositionalParse:
    values: dict[str, Any] = {}
    idx = 0

    for unit in positional_units:
        pf = unit.field if isinstance(unit, _SinglePositional) else unit.first
        if idx >= len(tokens):
            raise _ParseError(
                f"missing positional field '{pf.name}' ({_expected_value(pf)})"
            )

        token = tokens[idx]
        if token == "":
            raise _ParseError(f"empty token where field '{pf.name}' was expected")

        if isinstance(unit, _GroupedPositionals):
            first, second = _parse_grouped_positionals(token, unit)
            values[unit.first.name] = first
            values[unit.second.name] = second
        else:
            values[unit.field.name] = _parse_single_positional(token, unit.field)
        idx += 1

    return _PositionalParse(values=values, remaining=tokens[idx:])


def _parse_option_token(token: str, opt: _OptionField) -> _TokenParse | None:
    if not token.startswith(opt.tag):
        return None

    raw = token[len(opt.tag) :]
    owner = f"option '-{opt.tag}'"

    if raw == "":
        if opt.bare_value is not _MISSING:
            return _TokenParse(name=opt.name, value=opt.bare_value)
        raise _ParseError(f"option '-{opt.tag}' requires a {opt.codec.name} value")

    value = _parse_scalar_value(
        raw=raw,
        owner=owner,
        codec=opt.codec,
        choices=opt.choices,
    )
    return _TokenParse(name=opt.name, value=value)


def _parse_remaining(tokens: list[str], schema: _Schema) -> dict[str, Any]:
    values: dict[str, Any] = {}
    used_options: set[str] = set()
    used_flags: set[str] = set()

    for token in tokens:
        if token == "":
            raise _ParseError("empty token in tagged section")

        flag_value = schema.flags_by_tag.get(token)
        if flag_value is not None:
            if flag_value.name in used_flags:
                raise _ParseError(f"duplicate flag '-{token}'")
            values[flag_value.name] = True
            used_flags.add(flag_value.name)
            continue

        parsed_option: tuple[_OptionField, _TokenParse] | None = None
        for opt in schema.options_by_tag_length_desc:
            if not token.startswith(opt.tag):
                continue
            result = _parse_option_token(token, opt)
            if result is not None:
                parsed_option = (opt, result)
                break

        if parsed_option is None:
            expected = ", ".join(f"-{tag}" for tag in schema.known_tags)
            if expected:
                raise _ParseError(
                    f"unknown token {token!r}; expected one of: {expected}"
                    f"{_tag_suggestion(token, schema.known_tags)}"
                )
            raise _ParseError(f"unexpected extra token {token!r}")

        opt, result = parsed_option
        if opt.name in used_options:
            raise _ParseError(f"duplicate option '-{opt.tag}'")
        values[result.name] = result.value
        used_options.add(opt.name)

    return values


def _parse_from_schema(cls: type[Any], name: str, schema: _Schema) -> Any:
    if not isinstance(name, str):
        raise _ParseError(f"experiment name must be str; got {type(name).__name__}")

    tokens = _split_name(name)
    result = _parse_positionals(tokens, schema.positional_units)
    tagged_values = _parse_remaining(result.remaining, schema)

    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name in result.values:
            kwargs[f.name] = result.values[f.name]
        elif f.name in tagged_values:
            kwargs[f.name] = tagged_values[f.name]

    try:
        spec = cls(**kwargs)  # type: ignore[call-arg]
        spec.validate()
    except (TypeError, ValueError) as exc:
        raise _ParseError(str(exc)) from None
    return spec


# ---------------------------------------------------------------------------
# Name generation and help text
# ---------------------------------------------------------------------------


def _serialize_positionals(
    values: Mapping[str, Any],
    positional_units: tuple[_PositionalUnit, ...],
) -> list[str]:
    parts: list[str] = []
    for unit in positional_units:
        if isinstance(unit, _GroupedPositionals):
            first = _serialize_checked(
                owner=f"field '{unit.first.name}'",
                value=values[unit.first.name],
                codec=unit.first.codec,
            )
            second = _serialize_checked(
                owner=f"field '{unit.second.name}'",
                value=values[unit.second.name],
                codec=unit.second.codec,
            )
            parts.append(f"{first}{second}")
            continue

        pf = unit.field
        rendered = _serialize_checked(
            owner=f"field '{pf.name}'",
            value=values[pf.name],
            codec=pf.codec,
        )
        if pf.prefix is not None:
            rendered = f"{pf.prefix}{rendered}"
            if not _is_safe_atom(rendered):
                raise ValueError(
                    f"field '{pf.name}' with prefix serializes to invalid atom {rendered!r}"
                )
        parts.append(rendered)
    return parts


def _serialize_option(value: Any, opt: _OptionField) -> str:
    if opt.bare_value is not _MISSING and value == opt.bare_value:
        return opt.tag
    suffix = _serialize_checked(
        owner=f"option '-{opt.tag}'",
        value=value,
        codec=opt.codec,
    )
    rendered = f"{opt.tag}{suffix}"
    if not _is_safe_atom(rendered):
        raise ValueError(f"option '-{opt.tag}' serializes to invalid atom {rendered!r}")
    return rendered


def _format_default(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def _usage_token(pf: _PositionalField) -> str:
    if pf.prefix is not None:
        return f"{pf.prefix}<{pf.name}>"
    return f"<{pf.name}>"


def _usage(schema: _Schema) -> str:
    positional_parts: list[str] = []
    for unit in schema.positional_units:
        if isinstance(unit, _GroupedPositionals):
            positional_parts.append(f"<{unit.first.name}><{unit.second.name}>")
            continue
        positional_parts.append(_usage_token(unit.field))

    tagged_parts: list[str] = []
    for fl in sorted(schema.flags, key=lambda f: f.tag):
        tagged_parts.append(f"[-{fl.tag}]")
    for opt in sorted(schema.options, key=lambda o: o.tag):
        if opt.bare_value is _MISSING:
            tagged_parts.append(f"[-{opt.tag}<{opt.codec.name}>]")
        else:
            tagged_parts.append(f"[-{opt.tag}[<{opt.codec.name}>]]")

    return "-".join([*positional_parts, *tagged_parts]) or "<empty>"


def _help_text(spec_name: str, schema: _Schema) -> str:
    lines = [
        spec_name,
        f"Usage: {_usage(schema)}",
        "Grammar: '-' separated tokens; ordinary values cannot contain '-'; "
        "float exponent hyphens such as 1e-05 are allowed.",
    ]

    if schema.positional_units:
        lines.extend(["", "Positionals:"])
        for unit in schema.positional_units:
            if isinstance(unit, _GroupedPositionals):
                details = [
                    "one token",
                    f"{unit.first.name} "
                    f"{_choices_text(unit.first.choices) or unit.first.codec.name}",
                    f"{unit.second.name} "
                    f"{_choices_text(unit.second.choices) or unit.second.codec.name}",
                ]
                lines.append(
                    f"  {unit.first.name}+{unit.second.name}: "
                    + "; ".join(details)
                )
                continue

            pf = unit.field
            details = [pf.codec.name]
            if pf.prefix is not None:
                details.append(f"prefix: {pf.prefix!r}")
            if pf.choices:
                details.append(_choices_text(pf.choices))
            lines.append(f"  {pf.name}: " + "; ".join(details))

    if schema.flags:
        lines.extend(["", "Flags:"])
        for fl in sorted(schema.flags, key=lambda f: f.tag):
            lines.append(f"  -{fl.tag}: {fl.name}")

    if schema.options:
        lines.extend(["", "Options:"])
        for opt in sorted(schema.options, key=lambda o: o.tag):
            details = [opt.codec.name]
            if opt.choices:
                details.append(_choices_text(opt.choices))
            if opt.absent_default is not _MISSING:
                details.append(f"default: {_format_default(opt.absent_default)}")
                if not opt.emit_default:
                    details.append("default omitted from canonical name")
            if opt.bare_value is not _MISSING:
                details.append(f"bare: {_format_default(opt.bare_value)}")
            lines.append(f"  -{opt.tag}<{opt.codec.name}>: {opt.name}; " + "; ".join(details))

    if schema.examples:
        lines.extend(["", "Examples:"])
        for example in schema.examples:
            lines.append(f"  {example}")

    return "\n".join(lines)


def _validate_examples(cls: type[Any], schema: _Schema) -> None:
    for example in schema.examples:
        try:
            spec = _parse_from_schema(cls, example, schema)
            canonical = spec.to_name()
            reparsed = _parse_from_schema(cls, canonical, schema)
        except _ParseError as exc:
            raise TypeError(
                f"{cls.__name__} example {example!r} is invalid: {exc.reason}"
            ) from None
        except ValueError as exc:
            raise TypeError(
                f"{cls.__name__} example {example!r} is invalid: {exc}"
            ) from None

        if reparsed != spec:
            raise TypeError(
                f"{cls.__name__} example {example!r} canonicalizes to "
                f"{canonical!r} but does not round-trip"
            )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass_transform(field_specifiers=(positional, option, flag))
class ExperimentSpec:
    """Base class for declarative experiment-name specs."""

    _schema: ClassVar[_Schema]

    def __init_subclass__(
        cls,
        *,
        examples: Sequence[str] = (),
        frozen: bool = False,
        order: bool = False,
        unsafe_hash: bool = False,
        kw_only: bool = False,
        slots: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if slots:
            raise TypeError(
                "ExperimentSpec does not support slots=True via class keywords; "
                "dataclasses.dataclass(slots=True) returns a replacement class."
            )

        if "__dataclass_fields__" not in cls.__dict__:
            decorate = cast(
                Callable[[type[Any]], type[Any]],
                dataclasses.dataclass(
                    frozen=frozen,
                    order=order,
                    unsafe_hash=unsafe_hash,
                    kw_only=kw_only,
                ),
            )
            decorate(cls)

        if isinstance(examples, (str, bytes)):
            raise TypeError(
                f"{cls.__name__} examples must be a sequence of strings, not a string"
            )
        examples_tuple = tuple(examples)
        if not all(isinstance(example, str) for example in examples_tuple):
            raise TypeError(f"{cls.__name__} examples must be strings")

        schema = _build_schema(cls, examples_tuple)
        cls._schema = schema
        _validate_examples(cls, schema)

    @classmethod
    def try_parse(cls, name: str) -> Self | None:
        """Parse ``name`` into this spec, returning ``None`` if invalid."""
        try:
            return _parse_from_schema(cls, name, cls._schema)
        except _ParseError:
            return None

    @classmethod
    def parse(cls, name: str) -> Self:
        """Parse ``name`` into this spec or raise ``ExperimentParseError``."""
        try:
            return _parse_from_schema(cls, name, cls._schema)
        except _ParseError as exc:
            raise ExperimentParseError(
                spec_name=cls.__name__,
                name=name,
                reason=exc.reason,
                help_text=cls.help_text(),
            ) from None

    @classmethod
    def parse_or_raise(cls, name: str) -> Self:
        """Alias for ``parse()``."""
        return cls.parse(name)

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Alias for ``parse()``."""
        return cls.parse(name)

    @classmethod
    def help_text(cls) -> str:
        """Return schema-derived help text for this experiment spec."""
        return _help_text(cls.__name__, cls._schema)

    @classmethod
    def help(cls) -> str:
        """Alias for ``help_text()``."""
        return cls.help_text()

    @classmethod
    def register_codec(cls, python_type: type[T], codec: Codec[T]) -> None:
        """Register a codec. This is equivalent to the top-level function."""
        register_codec(python_type, codec)

    def validate(self) -> None:
        """Validate current field values against this spec's experiment schema."""
        schema = type(self)._schema
        values = {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

        for unit in schema.positional_units:
            fields_to_validate = (
                (unit.first, unit.second)
                if isinstance(unit, _GroupedPositionals)
                else (unit.field,)
            )
            for pf in fields_to_validate:
                _validate_value(
                    owner=f"{type(self).__name__}.{pf.name}",
                    value=values[pf.name],
                    codec=pf.codec,
                    choices=pf.choices,
                )

        for fl in schema.flags:
            value = values.get(fl.name, False)
            if type(value) is not bool:
                raise ValueError(
                    f"{type(self).__name__}.{fl.name} must be a bool; "
                    f"got {type(value).__name__}"
                )

        for opt in schema.options:
            value = values.get(opt.name)
            if value is None:
                continue
            _validate_value(
                owner=f"{type(self).__name__}.{opt.name}",
                value=value,
                codec=opt.codec,
                choices=opt.choices,
            )

    def to_name(self) -> str:
        """Generate the canonical experiment name string."""
        self.validate()
        schema = type(self)._schema
        values = {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

        parts = _serialize_positionals(values, schema.positional_units)

        tagged: list[tuple[str, str]] = []
        for fl in schema.flags:
            if values.get(fl.name, False):
                tagged.append((fl.tag, fl.tag))

        for opt in schema.options:
            value = values.get(opt.name)
            if value is None:
                continue
            if (
                not opt.emit_default
                and opt.absent_default is not _MISSING
                and value == opt.absent_default
            ):
                continue
            tagged.append((opt.tag, _serialize_option(value, opt)))

        tagged.sort(key=lambda item: item[0])
        parts.extend(rendered for _, rendered in tagged)
        return "-".join(parts)

    def name(self) -> str:
        """Backward-compatible alias for ``to_name()``."""
        return self.to_name()
