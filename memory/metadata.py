"""\
Defines metadata schema.\
Metadata is consisted of tags and will be attached to event for indexing.
"""

from pydantic import BaseModel, ValidationError
from typing import List, Any


class Tag(BaseModel):
    """现在只使用了str类型的value。更多数据类型的后面再实现，看是不是再加一个dtype字段"""

    key: str
    value: Any
    description: str = ""


class Metadata(BaseModel):
    tags: List[Tag] = []

    def __add__(self, other: "Metadata") -> "Metadata":
        return Metadata(tags=self.tags + other.tags)

    def __getitem__(self, indices):
        return self.tags.__getitem__(indices)

    def get(self, key: str, fallback=None):
        for tag in self.tags:
            if tag.key == key:
                return tag
        return fallback

    def update(self, key: str, value: str, upsert=False) -> Tag:
        tag = self.get(key=key)
        if tag:
            tag.value = value
            return tag
        else:
            if upsert:
                return self.insert(key=key, value=value)
            else:
                raise ValidationError(
                    "No matching key found while upsert flag is false."
                )

    def insert(self, key: str, value: str, description: str = "") -> Tag:
        if self.get(key=key):
            raise ValidationError("Same key tag exists.")
        tag = Tag(key=key, value=value, description=description)
        self.tags.append(tag)
        return tag
