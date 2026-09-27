from typing import Any


class RenderContext(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val
