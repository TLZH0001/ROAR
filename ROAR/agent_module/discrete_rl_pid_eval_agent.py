from ROAR.agent_module.agent import Agent
from pathlib import Path

from ROAR.control_module.discrete_rl_pid_eval_controller import PIDEvalController
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.configurations.configuration import Configuration as AgentConfig
import logging

from ROAR.planning_module.local_planner.waypoint_and_wayline_following_local_planner import SimpleWpAndWlFollowingLocalPlanner


class PIDEvalAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, target_speed = 40):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.target_speed = target_speed
        self.logger = logging.getLogger("PID Agent")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path) 
        
        self.pid_controller = PIDEvalController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWpAndWlFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5)

        self.logger.debug(
            f"Waypoint Following Agent Initiated. Reading f"
            f"rom {self.route_file_path.as_posix()}")

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(PIDEvalAgent, self).run_step(vehicle=vehicle,
                                       sensors_data=sensors_data)
        self.transform_history.append(self.vehicle.transform)
        control = self.local_planner.run_in_series()
        return control
