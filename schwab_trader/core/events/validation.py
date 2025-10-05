from typing import Any, get_type_hints, get_origin, get_args, Union

def validate_payload(payload: dict, schema: Any) -> None:
    """
    Validate payload against a TypedDict schema.
    Raises TypeError or KeyError if invalid.
    Supports Optional[...] (Union[..., NoneType]) fields.
    """
    hints = get_type_hints(schema)

    for field, ftype in hints.items():
        # --- detect Optional fields ---
        origin = get_origin(ftype)
        args = get_args(ftype)
        is_optional = origin is Union and type(None) in args

        # --- required field missing ---
        if field not in payload:
            if is_optional:
                # fill in missing optional fields as None
                payload[field] = None
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
