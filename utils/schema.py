from typing_extensions import Annotated
from datetime import datetime, time, date
from pydantic import PlainSerializer, BaseModel
from typing import Literal, Union
from enum import Enum
from zoneinfo import ZoneInfo
from utils.common import DEFAULT_AREA

# Note: defaults are handled by Pydantic Field() when used in model fields,
# not by Annotated metadata (Field inside Annotated is ignored by Pydantic V2)
Time = Annotated[
    time,
    PlainSerializer(lambda v, _: v.isoformat() if v else None),
]

Date = Annotated[
    date,
    PlainSerializer(lambda v, _: v.isoformat() if v else None),
]

DateTime = Annotated[
    datetime,
    PlainSerializer(lambda v, _: v.strftime('%Y-%m-%d %H:%M:%S') if v else None),
]

DateTimeString = Annotated[
    datetime,
    PlainSerializer(lambda v, _: v.timestamp() if v else None),
]

# 一些消息Content类型的Schema
class TextContent(BaseModel):
    msg_type: Literal["text"] = "text"
    content: str


class RichTextSyntaxEnum(str, Enum):
    markdown = "Markdown"
    html = "HTML"


class RichTextContent(BaseModel):
    msg_type: Literal["rich_text"] = "rich_text"
    syntax: RichTextSyntaxEnum = RichTextSyntaxEnum.markdown
    content: str


MessageContent = Union[
    TextContent,
    RichTextContent,
]
