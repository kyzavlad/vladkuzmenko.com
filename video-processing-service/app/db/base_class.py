from typing import Any
from sqlalchemy.ext.declarative import as_declarative, declared_attr
import inflect

# Initialize inflect engine for pluralization
p = inflect.engine()


@as_declarative()
class Base:
    """
    Base class for all database models.
    
    Provides common functionality:
    - Automatic __tablename__ generation
    - Common methods for serialization
    """
    id: Any
    __name__: str
    
    # Generate __tablename__ automatically based on class name
    @declared_attr
    def __tablename__(cls) -> str:
        # Convert CamelCase to snake_case and pluralize
        # Example: UserProfile -> user_profiles
        name = cls.__name__
        # Insert underscore between lowercase and uppercase letters
        # CamelCase -> Camel_Case
        import re
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name)
        # Convert to lowercase and pluralize
        return p.plural(name.lower())
    
    def to_dict(self) -> dict:
        """
        Convert model instance to a dictionary.
        Useful for API responses.
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if value is not None:
                result[column.name] = value
        return result
    
    def update_from_dict(self, data: dict) -> None:
        """
        Update model instance from a dictionary.
        Only updates columns that exist in the model.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value) 