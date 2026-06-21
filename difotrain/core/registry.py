"""A tiny name -> factory registry so robots, policies and data sources are
pluggable. Custom robots/policies register themselves and become available to
the CLI and config files without touching framework code.
"""
from __future__ import annotations

from typing import Any, Callable, Dict


class _Registry:
    def __init__(self) -> None:
        self.robots: Dict[str, Callable[..., Any]] = {}
        self.policies: Dict[str, Callable[..., Any]] = {}
        self.sources: Dict[str, Callable[..., Any]] = {}

    def _add(self, table: Dict[str, Callable], name: str, factory: Callable) -> None:
        if name in table:
            raise KeyError(f"'{name}' is already registered")
        table[name] = factory

    def make_robot(self, name: str, **kwargs) -> Any:
        return self._get(self.robots, "robot", name)(**kwargs)

    def make_policy(self, name: str, **kwargs) -> Any:
        return self._get(self.policies, "policy", name)(**kwargs)

    def make_source(self, name: str, **kwargs) -> Any:
        return self._get(self.sources, "source", name)(**kwargs)

    @staticmethod
    def _get(table: Dict[str, Callable], kind: str, name: str) -> Callable:
        try:
            return table[name]
        except KeyError:
            raise KeyError(
                f"unknown {kind} '{name}'. Registered: {sorted(table)}"
            ) from None


registry = _Registry()


def _decorator(table: Dict[str, Callable], name: str):
    def wrap(factory: Callable) -> Callable:
        registry._add(table, name, factory)
        return factory

    return wrap


def register_robot(name: str):
    return _decorator(registry.robots, name)


def register_policy(name: str):
    return _decorator(registry.policies, name)


def register_source(name: str):
    return _decorator(registry.sources, name)
