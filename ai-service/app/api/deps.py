"""Shared FastAPI dependencies for the AI service."""

from typing import Annotated

from fastapi import Depends

from ..config import Settings, get_settings

#: Injecting settings (rather than calling ``get_settings()`` inline) lets tests
#: override configuration per-app without touching the module-level cache.
SettingsDep = Annotated[Settings, Depends(get_settings)]
