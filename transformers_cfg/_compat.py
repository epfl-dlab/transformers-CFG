"""Backward compatibility utilities for deprecated names."""

import warnings
from functools import wraps


def deprecated_parameter(old_name: str, new_name: str, removal_version: str = "0.4.0"):
    """
    Mark a parameter name as deprecated.

    Args:
        old_name: Deprecated parameter name
        new_name: New parameter name to use instead
        removal_version: Version where the old name will be removed
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                warnings.warn(
                    f"Parameter '{old_name}' is deprecated and will be removed in version {removal_version}. "
                    f"Use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Map old name to new name
                if new_name not in kwargs:
                    kwargs[new_name] = kwargs.pop(old_name)
                else:
                    # Both old and new specified - prefer new
                    kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_attribute(old_name: str, new_name: str, removal_version: str = "0.4.0"):
    """
    Create a property that warns about deprecated attribute access.

    This returns a property descriptor that can be used in classes.

    Args:
        old_name: Deprecated attribute name
        new_name: New attribute name to use instead
        removal_version: Version where the old name will be removed

    Returns:
        Property descriptor that provides deprecated access

    Example:
        class MyClass:
            def __init__(self):
                self._parsing_stacks = []

            # Old name (deprecated)
            stacks = deprecated_attribute("stacks", "parsing_stacks")

            # New name
            @property
            def parsing_stacks(self):
                return self._parsing_stacks
    """

    def getter(self):
        warnings.warn(
            f"Attribute '{old_name}' is deprecated and will be removed in version {removal_version}. "
            f"Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)

    def setter(self, value):
        warnings.warn(
            f"Attribute '{old_name}' is deprecated and will be removed in version {removal_version}. "
            f"Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(self, new_name, value)

    return property(getter, setter)


def deprecated_method(new_name: str, removal_version: str = "0.4.0"):
    """
    Mark a method as deprecated and redirect to new method.

    Args:
        new_name: New method name to use instead
        removal_version: Version where the old method will be removed

    Example:
        class MyClass:
            @deprecated_method("can_accept_bytes")
            def try_accept_bytes(self, data):
                return self.can_accept_bytes(data)

            def can_accept_bytes(self, data):
                # Actual implementation
                pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            warnings.warn(
                f"Method '{func.__name__}' is deprecated and will be removed in version {removal_version}. "
                f"Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
