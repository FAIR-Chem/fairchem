from logging import getLogger

from typing_extensions import Self, override

log = getLogger(__name__)


class Singleton:
    singleton_key = "_singleton_instance"

    @classmethod
    def get(cls) -> Self | None:
        return getattr(cls, cls.singleton_key, None)

    @classmethod
    def set(cls, instance: Self) -> None:
        if cls.get() is not None:
            log.warning(f"{cls.__qualname__} instance is already set")

        setattr(cls, cls.singleton_key, instance)

    @classmethod
    def reset(cls) -> None:
        if cls.get() is not None:
            delattr(cls, cls.singleton_key)

    @classmethod
    def register(cls, instance: Self) -> None:
        cls.set(instance)

    @classmethod
    def instance(cls) -> Self:
        instance = cls.get()
        if instance is None:
            raise RuntimeError(f"{cls.__qualname__} instance is not set")

        return instance

    @override
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)

        cls.reset()
