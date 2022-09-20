#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla

from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """
    sensor_data = {}
    reward = {}

    def __init__(self, path_to_conf_file):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        # 这个接口全局只能申明一次，否则会产生sensor之间的冲突
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file)

        self.wallclock_t0 = None

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass
    
    def get_sensor_data(self):
        # self.sensor_data = self.sensor_interface.get_data()
        return self.sensor_data 
    # 调用autonomous_agent实例时，会返回这个control
    # 子类继承时，会首先执行父类的方法，此方法也会被执行
    def __call__(self, control=None, no_model=False):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        # 有一个异步线程在存取sensor_interface_data，用户不能自己重复读
        input_data = self.sensor_interface.get_data()
        self.sensor_data = input_data
        # 所以会输出它吗？——会！
        print("----------------------")
        # print("input_sensor_data(before action):\n",input_data)

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        # print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))

        # 可以走定制的control
        if no_model:
            control=carla.libcarla.VehicleControl(control)
        else:
            control = self.run_step(input_data, timestamp)


        control.manual_gear_shift = False

        # 经过run_step，也就是已经apply_control了，可以返回此时的reward, sensor_info
        # obs返回不了，不能随时读取sensor_data
        
        # record_score=statistics_manager.compute_route_statistics(config)
        # action_reward=record_score.scores
        # print("reward after action:",action_reward)
        # self.reward = action_reward
        # obs =  self.sensor_interface.get_data()
        # self.sensor_data = obs

        
        # print("---------------------------")
        # print("type(control):",type(control))
        # print("control:",control)
        # print("---------------------------")
 
#         control=carla.libcarla.VehicleControl(throttle=0.750000, steer=0.000000, brake=0.000000, 
# hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)

        return control, input_data

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
