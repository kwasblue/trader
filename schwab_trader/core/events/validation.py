from typing import Any, get_type_hints, get_origin, get_args, Union

def validate_payload(payload: dict, schema: Any) -> None:
    """
    Validate payload against a TypedDict schema.
    Raises TypeError or KeyError if invalid.
    Supports Optional[...] (Union[..., NoneType]) fields.
    Respects total=False (all fields optional) on TypedDict.
    """
    hints = get_type_hints(schema)

    # Check if TypedDict has total=False (all fields optional)
    # TypedDict with total=False has __total__ = False
    all_optional = getattr(schema, '__total__', True) is False

    for field, ftype in hints.items():
        # --- detect Optional fields ---
        origin = get_origin(ftype)
        args = get_args(ftype)
        is_optional = origin is Union and type(None) in args

        # --- required field missing ---
        if field not in payload:
            # Field is optional if: explicitly Optional[X], OR total=False on TypedDict
            if is_optional or all_optional:
                # Skip missing optional fields (don't mutate payload)
                continue
            else:
                raise KeyError(f"Missing required field '{field}' for {schema.__name__}")

        val = payload[field]
        if val is None:
            continue  # allow None, Optional handled by type checker

        # --- lightweight runtime type check ---
        try:
            # handle Optional and Union types
            if origin is Union:
                valid_types = tuple(t for t in args if t is not type(None))
                if not isinstance(val, valid_types):
                    raise TypeError(f"Field '{field}' must be one of {valid_types}, got {type(val)}")
            elif not isinstance(val, ftype):
                raise TypeError(f"Field '{field}' must be {ftype}, got {type(val)}")
        except TypeError:
            # some typing constructs (e.g., Literal, Annotated) may break isinstance()
            pass
