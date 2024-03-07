from pathlib import Path
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, validator, root_validator, PositiveInt, Field
from enum import Enum

from utils.path import PERSONA_DB_DIR
from observation.schedule_module import DailySchedule
from memory.event import ExperienceEvent


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"
    prefer_not_to_say = "Prefer not to say"


class Height(BaseModel):
    feet: Optional[int] = None
    inches: Optional[int] = None
    unit: Literal['cm', 'inch', 'foot']  # The unit of the height value
    height: Optional[Union[float, int]] = None  # The height value in the base unit, can be None initially

    @root_validator(pre=True)
    def calculate_height(cls, values):
        feet = values.get('feet')
        inches = values.get('inches')
        height = values.get('height')
        unit = values.get('unit')

        if unit in ['inch', 'foot'] and (feet is not None or inches is not None):
            # Convert feet and inches to inches
            total_inches = (feet or 0) * 12 + (inches or 0)
            if unit == 'inch':
                height = total_inches
            else:  # unit == 'foot'
                height = total_inches / 12
        elif unit == 'cm' and height is not None:
            # Height is already in cm, no need to convert
            pass
        else:
            raise ValueError("Invalid height or unit provided")

        values['height'] = height
        return values

    @property
    def in_cm(self) -> float:
        if self.unit == 'cm':
            return self.height # type: ignore
        elif self.unit == 'inch':
            return self.height * 2.54 # type: ignore
        else:  # self.unit == 'foot'
            return self.height * 30.48 # type: ignore

    @property
    def in_feet_and_inches(self) -> str:
        if self.unit == 'cm':
            total_inches = self.height / 2.54 # type: ignore
        else:  # self.unit in ['inch', 'foot']
            total_inches = self.height * 12 if self.unit == 'foot' else self.height # type: ignore
        
        feet = int(total_inches // 12) # type: ignore
        inches = int(total_inches % 12) # type: ignore
        return f"{feet}' {inches}\""

    @validator('height')
    def height_must_be_positive(cls, value):
        assert value is None or value > 0, 'Height must be a positive number'
        return value


class Weight(BaseModel):
    weight: Union[float, int]  # The weight value in the base unit of kg
    unit: Literal['kg', 'pound', 'jin']  # The unit of the weight value

    @property
    def in_kg(self) -> float:
        if self.unit == 'kg':
            return self.weight
        elif self.unit == 'pound':
            return self.weight * 0.45359237
        else:  # self.unit == 'jin'
            return self.weight * 0.5

    @property
    def in_pounds(self) -> float:
        if self.unit == 'pound':
            return self.weight
        elif self.unit == 'kg':
            return self.weight / 0.45359237
        else:  # self.unit == 'jin'
            return self.weight * 2 * 0.45359237

    @property
    def in_jin(self) -> float:
        if self.unit == 'jin':
            return self.weight
        elif self.unit == 'kg':
            return self.weight * 2
        else:  # self.unit == 'pound'
            return self.weight / 0.45359237 / 2

    @validator('weight')
    def weight_must_be_positive(cls, value):
        assert value > 0, 'Weight must be a positive number'
        return value
    

class MBTIType(Enum):
    INTP = "INTP"
    INTJ = "INTJ"
    INFP = "INFP"
    INFJ = "INFJ"
    ISTP = "ISTP"
    ISTJ = "ISTJ"
    ISFP = "ISFP"
    ISFJ = "ISFJ"
    ENTP = "ENTP"
    ENTJ = "ENTJ"
    ENFP = "ENFP"
    ENFJ = "ENFJ"
    ESTP = "ESTP"
    ESTJ = "ESTJ"
    ESFP = "ESFP"
    ESFJ = "ESFJ"


class MBTI(BaseModel):
    mbti_type: MBTIType
    trait: Optional[Literal['A', 'T']] = None

    @validator('mbti_type', pre=True)
    def uppercase_mbti_type(cls, value):
        if isinstance(value, str):
            value = value.upper()  # 将输入字符串转换为大写
        return value

    @property
    def full_mbti(self) -> str:
        """
        Returns the full MBTI result including the optional trait.
        Example: 'intp' with trait 'A' will be returned as 'INTP-A'.
        """
        mbti_result = self.mbti_type.value
        if self.trait:
            mbti_result += f"-{self.trait}"
        return mbti_result
    

class LikesAndDislikes(BaseModel):
    likes: List[str] = []
    dislikes: List[str] = []


class RelationType(str, Enum):
    FRIEND = "Friend"
    SPOUSE = "Spouse"
    COLLEAGUE = "Colleague"
    SIBLING = "Sibling"
    PARENT = "Parent"
    CHILD = "Child"
    RELATIVE = "Relative"
    PARTNER = "Partner"
    MENTOR = "Mentor"
    MENTEE = "Mentee"
    NEIGHBOR = "Neighbor"
    ROOMMATE = "Roommate"
    CLASSMATE = "Classmate"
    TEAMMATE = "Teammate"
    OTHER = "Other"


class Relationship(BaseModel):
    name: str
    relation: RelationType
    details: Optional[str]


class Persona(BaseModel):
    name: str
    age: Optional[PositiveInt] = Field(description='Age of the character.', default=None)
    gender: Optional[GenderEnum] = Field(description='Gender of the character', default=None)
    height: Optional[Height] = Field(description='Height of the character.', default=None)
    weight: Optional[Weight] = Field(description='Weight of the character.', default=None)
    introduction: str = Field(description='Introduction of the character.', default="")
    personality: str = Field(description='Personality of the character.', default="")
    mbti: Optional[MBTI] = Field(description='MBTI of the character.', default=None)
    likes_and_dislikes: Optional[LikesAndDislikes] = Field(description='Likes and dislikes of the character.', default=None)
    relationship: Optional[List[Relationship]] = Field(description='Relationship of the character.', default=None)
    regular_schedule: Optional[DailySchedule] = Field(description='Regular schedule of the agent.', default=None)
    story: Optional[List[ExperienceEvent]] = Field(description='Story of the character.', default=None)

    def prompt(self, 
               indent=4, 
               fields: Optional[List[Literal[
                                          'name',
                                          'age',
                                          'gender',
                                          'height',
                                          'weight',
                                          'introduction', 
                                          'personality', 
                                          'mbti', 
                                          'likes_and_dislikes',    
                                          'relationship', 
                                          'regular_schedule',
                                          'story']]] = None, 
                include=False) -> str:
        fields_set = set(fields) if fields is not None else set()

        if include:
            # 只包含指定字段
            return self.model_dump_json(include=fields_set, indent=indent)
        else:
            # 排除指定字段
            return self.model_dump_json(exclude=fields_set, indent=indent)

    @classmethod
    def from_json(cls, file: str, directory: Path = PERSONA_DB_DIR) -> 'Persona':
        with open(directory / file, "r") as f:
            return cls.model_validate_json(f.read())
        
    def to_json(self, file: str, directory: Path = PERSONA_DB_DIR) -> None:
        with open(directory / file, "w") as f:
            f.write(self.model_dump_json(indent=4))
