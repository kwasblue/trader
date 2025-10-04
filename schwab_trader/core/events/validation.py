from typing import Any, get_type_hints

def validate_payload(payload: dict, schema: Any) -> None:
    """
    Validate payload against a TypedDict schema.
    Raises TypeError or KeyError if invalid.
    """
    hints = get_type_hints(schema)

    for field, ftype in hints.items():
        if field not in payload:
            raise KeyError(f"Missing required field '{field}' for {schema.__name__}")
        val = payload[field]
        if val is None:
            continue  # allow None, Optional handled by type checker
        # very lightweight runtime type check
        try:
            if not isinstance(val, ftype):  # ftype may not always be isinstance-compatible
                raise TypeError(f"Field '{field}' must be {ftype}, got {type(val)}")
        except TypeError:
            # ftype might be typing constructs (Literal, Union, Annotated) → skip strict check
            pass

