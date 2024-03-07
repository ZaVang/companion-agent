from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Union, Callable
from altair import value
from pydantic import BaseModel, root_validator, ValidationError

from utils.path import *
from utils.schema import Date, Time, DateTime


class DailyEvent(BaseModel):
    start_time: Time
    end_time: Optional[Time] = None
    event: str
    duration: Optional[int] = None

    @root_validator(pre=False, skip_on_failure=True)
    def check_times_and_duration(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        duration = values.get('duration')

        if duration is not None and end_time is not None:
            calculated_duration = (datetime.combine(date.today(), end_time) - datetime.combine(date.today(), start_time)).total_seconds() // 60
            if calculated_duration != duration:
                raise ValueError('end_time does not match the duration from start_time')
        elif duration is not None:
            values['end_time'] = (datetime.combine(date.today(), start_time) + timedelta(minutes=duration)).time()
        elif end_time is not None:
            duration_seconds = (datetime.combine(date.today(), end_time) - datetime.combine(date.today(), start_time)).total_seconds()
            if duration_seconds < 0:
                raise ValueError('end_time is before start_time')
            values['duration'] = int(duration_seconds) // 60
            
        return values
    
    @property
    def text(self):
        if self.end_time is not None:
            return f"{self.start_time} - {self.end_time}: {self.event}"
        elif self.duration is not None:
            end_time = (datetime.combine(date.today(), self.start_time) + timedelta(minutes=self.duration)).time()
            return f"{self.start_time} - {end_time}: {self.event}"
        else:
            return f"{self.start_time}: {self.event}"


class DailySchedule(BaseModel):
    events: List[DailyEvent]

    @property
    def text(self):
        return "\n".join([e.text for e in self.events])


class EventInstance(BaseModel):
    start_time: DateTime
    event: str
    end_time: Optional[DateTime] = None
    duration: Optional[int] = None

    @root_validator(pre=False, skip_on_failure=True)
    def check_times_and_duration(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        duration = values.get('duration')

        if duration is not None and end_time is not None:
            calculated_duration = (end_time - start_time).total_seconds() // 60
            if calculated_duration != duration:
                raise ValueError('end_time does not match the duration from start_time')
        elif duration is not None:
            values['end_time'] = start_time + timedelta(minutes=duration)
        elif end_time is not None:
            duration_seconds = (end_time - start_time).total_seconds()
            if duration_seconds < 0:
                raise ValueError('end_time is before start_time')
            values['duration'] = int(duration_seconds) // 60
            
        return values

    @property
    def text(self):
        if self.end_time is not None:
            return f"{self.start_time} - {self.end_time}: {self.event}"
        elif self.duration is not None:
            end_time = self.start_time + timedelta(minutes=self.duration)
            return f"{self.start_time} - {end_time}: {self.event}"
        else:
            return f"{self.start_time}: {self.event}"


class Schedule(BaseModel):
    events: List[EventInstance]

    def __getitem__(self, indices):
        """可以用[x]/[x:y]/[datetime.date]/[datetime.date:datetime.date]索引"""
        if isinstance(indices, date):
            return Schedule(
                events=[e for e in self.events if e.start_time.date() == indices]
            )
        elif isinstance(indices, slice) and isinstance(indices.start, Date):
            return Schedule(
                events=[
                    e
                    for e in self.events
                    if e.start_time.date() >= indices.start
                    and e.start_time.date() <= indices.stop
                ]
            )
        else:
            if isinstance(indices, slice):
                return Schedule(events=self.events.__getitem__(indices))
            else:
                return self.events.__getitem__(indices)

    def sort(self, reverse=False):
        self.events = sorted(self.events, key=lambda x: x.start_time, reverse=reverse)
        return self

    def _group_by_date(self) -> Dict[date, List[EventInstance]]:
        groups = defaultdict(list)
        for e in self.events:
            groups[e.start_time.date()].append(e)
        return groups

    @classmethod
    def from_dailyschedule(
        cls, dailyschedule: DailySchedule, date: date = date.today()
    ) -> "Schedule":
        events = []
        for e in dailyschedule.events:
            _e = EventInstance(
                start_time=datetime.combine(date, e.start_time), event=e.event
            )
            events.append(_e)
        return cls(events=events)

    @property
    def text(self):
        self.sort()
        string, groups = "", self._group_by_date()
        for key in groups:
            string += str(key) + "\n"
            string += "\n".join(
                [e.text for e in groups[key]]
            )
            string += "\n"
        return string
    
    def _check_conflict(self, event: EventInstance) -> Optional[EventInstance]:
        """检查给定的事件是否与现有事件冲突"""
        for e in self.events:
            # 处理 end_time 为 None 的情况
            e_end_time = e.end_time if e.end_time else e.start_time
            event_end_time = event.end_time if event.end_time else event.start_time
            
            if (event.start_time <= e.start_time < event_end_time) or \
            (event.start_time < e_end_time <= event_end_time):
                return e
        return None

    def add_event(self, event: EventInstance, overwrite: bool = False):
        conflicting_event = self._check_conflict(event)
        if conflicting_event:
            if overwrite:
                self.remove_event(conflicting_event)
            else:
                raise ValueError(f"Time conflict with event: {conflicting_event.text}")
        self.events.append(event)
        self.sort()

    def remove_event(self, item):
        """可以用[x]/[x:y]/[datetime.date]/[datetime.date:datetime.date]或直接的EventInstance来删除事件"""
        if isinstance(item, EventInstance):
            if item in self.events:
                self.events.remove(item)
        elif isinstance(item, date):
            self.events = [e for e in self.events if e.start_time.date() != item]
        elif isinstance(item, slice) and isinstance(item.start, date):
            self.events = [
                e
                for e in self.events
                if not (e.start_time.date() >= item.start and e.start_time.date() <= item.stop)
            ]
        else:
            if isinstance(item, slice):
                to_remove = self.events.__getitem__(item)
                self.events = [e for e in self.events if e not in to_remove]
            else:
                event_to_remove = self.events.__getitem__(item)
                self.events.remove(event_to_remove)

    def filter_events(self, condition: Callable[[EventInstance], bool]) -> "Schedule":
        """
        根据给定的条件过滤事件
        
        Example:
        # 使用lambda函数过滤所有在某个日期之前开始的事件
        filtered_schedule = schedule.filter_events(lambda e: e.start_time.date() < date(2023, 1, 1))
        """
        return Schedule(events=[e for e in self.events if condition(e)])

    def to_json(self, file:str, directory: Path = SCHEDULE_DB_DIR) -> None:
        with open(directory / file, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json(cls, file: str, directory: Path = SCHEDULE_DB_DIR) -> "Schedule":
        with open(directory / file, "r") as f:
            return cls.model_validate_json(f.read())