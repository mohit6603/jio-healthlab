"""Database session dependency."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from ..database import get_db

get_db_session = get_db

#: Annotated alias so routers read as ``db: DbSession`` instead of repeating
#: ``Depends(get_db)`` everywhere.
DbSession = Annotated[Session, Depends(get_db)]
