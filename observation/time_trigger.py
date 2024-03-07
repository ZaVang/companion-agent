from typing import Dict, Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from functools import partial
from datetime import datetime

from utils.common import DEFAULT_AREA
from interaction.behavior import default_behavior
from observation.schedule_module import Schedule, DailySchedule
from observation.time_module import TimeModule

class TimeTrigger:    
    def __init__(self, timezone: str=DEFAULT_AREA):
        self.scheduler = BackgroundScheduler({'apscheduler.timezone':timezone})
        self.time_module = TimeModule(timezone)
    
    def add_schedule_job(self, dailyschedule: DailySchedule, behavior_list: Optional[List]=None):
        time_list = []
        action_list = []
        curr_time = self.time_module.get_current_time(is_str=False)

        for date, actions in dailyschedule.to_dict().items():
            for time, action in actions.items():
                # 构造完整的 datetime 对象
                time_str = f"{date} {time}"
                action_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
                action_time = self.time_module.timezone.localize(action_time)
                
                # 如果这个时间比当前时间晚，将它和相应的动作加入列表
                if action_time > curr_time:
                    time_list.append(action_time)
                    action_list.append(action)
        
        if behavior_list is not None:
            assert len(time_list) == len(behavior_list), 'length of time schedule and behavior list must be equal'
            for time, behavior, action in zip(time_list, behavior_list, action_list):
                job_id = str(self.time_module.to_timestamp(time))
                self.add_job(time, partial(behavior, time=time, action=action), job_id)
        else:
            for time, action in zip(time_list, action_list):
                job_id = str(self.time_module.to_timestamp(time))
                self.add_job(time, partial(default_behavior, time=time, action=action), job_id)
    
    def add_job(self, date, behavior, job_id):
        self.scheduler.add_job(behavior, 'date', run_date=date, id=job_id)
    
    def remove_job(self, job_id):
        self.scheduler.remove_job(job_id)