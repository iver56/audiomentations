from __future__ import annotations

import warnings
from abc import ABC, ABCMeta, abstractmethod
from typing import Any

SERIALIZABLE_REGISTRY: dict[str, SerializableMeta] = {}
NON_SERIALIZABLE_REGISTRY: dict[str, SerializableMeta] = {}


def shorten_class_name(class_fullname: str) -> str:
    split = class_fullname.split(".")
    if len(split) == 1:
        return class_fullname
    top_module, *_, class_name = split
    if top_module == "audiomentations":
        return class_name
    return class_fullname
    
class SerializableMeta(ABCMeta):
    """A metaclass that is used to register classes in `SERIALIZABLE_REGISTRY` or `NON_SERIALIZABLE_REGISTRY`
    so they can be found later while deserializing transformation pipeline using classes full names.
    """

    def __new__(cls, name: str, bases: tuple[type, ...], *args: Any, **kwargs: Any) -> SerializableMeta:
        cls_obj = super().__new__(cls, name, bases, *args, **kwargs)
        if name != "Serializable" and ABC not in bases:
            if cls_obj.is_serializable():
                SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
            else:
                NON_SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
        return cls_obj

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def _to_dict(cls) -> dict[str, Any]:
        return {}


class Serializable(metaclass=SerializableMeta):
    @classmethod
    @abstractmethod
    def is_serializable(cls) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_class_fullname(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_dict_private(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self, on_not_implemented_error: str = "raise") -> dict[str, Any]:
        """Take a transform pipeline and convert it to a serializable representation that uses only standard
        python data types: dictionaries, lists, strings, integers, and floats.

        Args:
            self: A transform that should be serialized. If the transform doesn't implement the `to_dict`
                method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
                If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
                but no transform parameters will be serialized.
            on_not_implemented_error (str): `raise` or `warn`.

        """
        if on_not_implemented_error not in {"raise", "warn"}:
            msg = f"Unknown on_not_implemented_error value: {on_not_implemented_error}. Supported values are: 'raise' "
            "and 'warn'"
            raise ValueError(msg)
        try:
            transform_dict = self.to_dict_private()
        except NotImplementedError:
            if on_not_implemented_error == "raise":
                raise

            transform_dict = {}
            warnings.warn(
                f"Got NotImplementedError while trying to serialize {self}. Object arguments are not preserved. ",
                stacklevel=2,
            )
        return {"transform": transform_dict}

def get_shortest_class_fullname(cls: type[Any]) -> str:
    """The function `get_shortest_class_fullname` takes a class object as input and returns its shortened
    full name.

    :param cls: The parameter `cls` is of type `Type[BasicCompose]`, which means it expects a class that
    is a subclass of `BasicCompose`
    :type cls: Type[BasicCompose]
    :return: a string, which is the shortened version of the full class name.
    """
    class_fullname = f"{cls.__module__}.{cls.__name__}"
    return shorten_class_name(class_fullname)
