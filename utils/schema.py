from typing_extensions import Annotated
from datetime import datetime, time, date
from pydantic import BeforeValidator ,PlainSerializer, Field, BaseModel
from typing import Literal, Union
from enum import Enum
from zoneinfo import ZoneInfo
from utils.common import DEFAULT_AREA

Time = Annotated[
    time,
    Field(default=time(0, 0, 0)),
]

Date = Annotated[
    date,
    Field(default=date(2023, 1, 1)),
]

##TODO：应该直接用一个pydantic类来定义
DateTime = Annotated[
    datetime,
    PlainSerializer(lambda v, _: v.strftime('%Y-%m-%d %H:%M:%S')),
    Field(default_factory=lambda: datetime.now(ZoneInfo(DEFAULT_AREA))),
]

DateTimeString = Annotated[
    datetime,
    PlainSerializer(lambda v, _: v.timestamp()),
    Field(default_factory=lambda: datetime.now(ZoneInfo(DEFAULT_AREA))),
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


MessageContent = Annotated[
    Union[
        TextContent,
        RichTextContent,
    ],
    Field(None, description="消息Content类型"),
]
