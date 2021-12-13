from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import Transform
from collections import deque
import numpy as np
import logging
from ROAR.agent_module.agent import Agent
from typing import Any, Tuple
import json
from pathlib import Path
from stable_baselines3 import DQN

from ROAR_Gym.Discrete_PID.valid_pid_action import MAX_SPEED, TARGET_SPEED, VALID_ACTIONS


turning_boxes = [   [803, 870, -630, -570], 
                    [788, 826, -356, -190],
                    [690, 732, 677, 706], 
                    [-620, -490, 642, 746], 
                    [-676, -720, -115, -2.5], 

                    [-843, -827.4, -405, -265],
                    [-850, -821, -686, -278] 
                    #[783, 865, -482, -198.3]

]


def turning(cur_loc):
    turning = False

    for i, box in enumerate(turning_boxes):
        if in_turning_box(cur_loc, box):
            turning = True
            #print(i)
            return turning

    return turning


def in_turning_box(cur_loc, box_boundary):
    x = cur_loc[0]
    z = cur_loc[1]
    x1 = box_boundary[0]
    x2 = box_boundary[1]
    z1 = box_boundary[2]
    z2 = box_boundary[3]
    #print(x, (x1, x2), (x >= x1 and x<=x2), z, (z1, z2),(z >= z1 and z <= z2))
    return (x >= x1 and x<=x2 and z >= z1 and z <= z2)




class LongPIDController(Controller):
    def __init__(self, agent, config: dict, throttle_boundary: Tuple[float, float], max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)

        self._dt = dt


    def run_in_series(self, next_waypoint: Transform, next_wayline, current_dir, **kwargs) -> float:
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        roll = self.agent.vehicle.transform.rotation.roll
        out = np.exp(-0.07 * np.abs(roll))
        output = float(np.clip(out, self.throttle_boundary[0], self.throttle_boundary[1]))

        if len(next_wayline) == 3 :
            slope1 = next_wayline["current_wayline"].slope
            slope2 = next_wayline["look_ahead_wayline"].slope
            slope3 = next_wayline["target_wayline"].slope
            tan1 = np.abs((slope1 - slope2) / (1 + slope1 * slope2))
            tan2 = np.abs((slope2 - slope3) / (1 + slope2 * slope3))
            tan3 = np.abs((slope1 - slope3) / (1 + slope1 * slope3))
            # tan1 indicate if agent is actually turning
            # tan2 indicate if there is a turn ahead
            # tan3 indicate if there is a turn far ahead
            
            if tan1 >= 1 and current_speed >= TARGET_SPEED:
                #print("low-speed turining")
                return self.throttle_boundary[0]

            if tan1 >= 0.5 and current_speed >= MAX_SPEED:
                #print("high-speed turning")
                return self.throttle_boundary[0]
           
            if tan3 >= 1 and current_speed > TARGET_SPEED:
                #print("sharp turn ahead")
                return self.throttle_boundary[0]
            #if tan2 >= 1:
            #    print("turning ahead")
            #    return self.throttle_boundary[0]
            #if tan1 >=0.5:
            #    print("turning")
            #    return self.throttle_boundary[0]
        elif len(next_wayline) == 2:
            return self.throttle_boundary[1]
        elif len(next_wayline) == 1:
            return self.throttle_boundary[1]

        return output

class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        #print(config)
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        print(current_speed, k_p, k_i, k_d)
        return np.array([k_p, k_d, k_i])

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p = self.agent.kwargs["lat_k_p"]
        k_d = self.agent.kwargs["lat_k_d"]
        k_i = self.agent.kwargs["lat_k_i"]

        lat_control = float(
              np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
             )

        # k_p, k_d, k_i = [0.2, 0.02, 0]
        # cal_steering = (k_p * error) + (k_d * _de) + (k_i * _ie)
        # lat_control =  0
        # #print(v_begin)
        
        # if turning((v_begin[0], v_begin[2])):
        #     #print(turning)
        #     lat_control = float(
        #      np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), -1, 1)
        #     )
        # else:
        #     k_p, k_d, k_i = [0.2, 0.02, 0]
        #     lat_control = float(
        #      np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        #     )


        #k_p, k_d, k_i = LatPIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)
        # print(self.agent.kwargs["lat_k_p"], self.agent.kwargs["lat_k_d"], self.agent.kwargs["lat_k_i"])
        # print(k_p, k_d, k_i)

      
        return lat_control

class PIDEvalController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        
        super().__init__(agent)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(self.agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.long_pid_controller = LongPIDController(agent=self.agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed,
                                                     config=self.config["longitudinal_controller"])
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

        self.init_flag = True
        # import os
        # print(os.getcwd())
        self.pid_rl_model = DQN.load(Path("./ROAR_Gym/output/discrete_pid_logs/rl_model_800000_steps"))
        # try:
        #     self.pid_rl_model = DQN.load(Path("../output/discrete_pid_logs/rl_model_1000000_steps.zip"))
        # except:
        #     print("error happens")
        #     path = Path(self.agent.kwargs['kwargs']["rl_pid_model_file_path"])
        #     self.pid_rl_model = DQN.load(load_path=path)

    def get_obs(self) -> Any:
        curr_speed = np.array([Vehicle.get_speed(self.agent.vehicle)])
        curr_transform = self.agent.vehicle.transform.to_array()
        # print(curr_transform)
        # if(curr_transform[5] == 0):
        #     curr_transform[5] = -90

        if len(self.agent.local_planner.way_points_queue) > 0:
            next_waypoint_transform = self.agent.local_planner.way_points_queue[0].to_array()
        else:
            next_waypoint_transform = curr_transform
        return np.append(np.append(curr_speed, curr_transform), next_waypoint_transform)

    def run_in_series(self, next_waypoint: Transform, next_wayline = None,  current_dir = None, **kwargs) -> VehicleControl:
        obs = self.get_obs()
        action, _ = self.pid_rl_model.predict(obs)
        action = VALID_ACTIONS[int(action)]

        lat_k_p, lat_k_d, lat_k_i = action[0], action[1], action[2]

        self.agent.kwargs["lat_k_p"] = lat_k_p
        self.agent.kwargs["lat_k_d"] = lat_k_d
        self.agent.kwargs["lat_k_i"] = lat_k_i

        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          next_wayline = next_wayline, 
                                                          current_dir = current_dir
                                                          )
     

        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint)
        

        control = VehicleControl(throttle=throttle, steering=steering)
        return control
