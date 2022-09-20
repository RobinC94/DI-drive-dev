from ast import arguments
import sys
from typing import Any, Dict
import carla
from gym import spaces
import numpy as np
import os
from easydict import EasyDict
import os
import time
import traceback
import signal
from core.policy.interfuser_policy import InterfuserPolicy
from leaderboard_package.team_code.planner import RoutePlanner
from leaderboard_package.team_code.utils import transform_2d_points,lidar_to_histogram_features
from leaderboard_package.leaderboard.utils.route_manipulation import downsample_route, interpolate_trajectory
import py_trees

from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.result_writer import ResultOutputProvider
from leaderboard.utils.route_indexer import RouteIndexer
from scenarios.route_scenario import convert_transform_to_location
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from leaderboard.scenarios.route_scenario import RouteScenario
from core.utils.simulator_utils.sensor_helper import SensorHelper
import carla
import pkg_resources
import cv2
import torch
from torchvision import transforms
from PIL import Image
from leaderboard_package.team_code.tracker import Tracker
from leaderboard_package.team_code.render import render, render_self_car, render_waypoints
from leaderboard_package.team_code.interfuser_controller import InterfuserController
from leaderboard_package.timm.models import create_model


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth':      'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.lidar.ray_cast_semantic':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}

class ScenarioLeaderboardEnv():

    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})
    reward_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # sensors
    # sensor_icons = []

    def __init__(
            self,
            cfg: Dict,
            host: str = 'localhost',
            port: int = 2000,
            tm_port: int = 2500,
            **kwargs,        ) -> None:
        self.statistics_manager = StatisticsManager()
        self.cfg = cfg
        # 需要改成绝对路径
        os.environ["ROUTES"] = self.cfg.arguments.routes
        os.environ["CHECKPOINT_ENDPOINT"] = self.cfg.arguments.checkpoint
        os.environ["SCENARIOS"] = self.cfg.arguments.scenarios
        os.environ["SAVE_PATH"] = self.cfg.arguments.savepath

        self.client = carla.Client(host, port)
        if self.cfg.arguments.timeout:
            self.client_timeout = float(self.cfg.arguments.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(self.cfg.arguments.trafficManagerPort))
        # self.hero_vehicle = None
        self._timestamp_last_run = 0.0
        self._init_carla_world()
        self.ego_vehicles = []
#         self.action = carla.libcarla.VehicleControl(throttle=0.0, steer=0.000000, brake=0.000000, \
# hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0

        self.crash_message = ""
        self.entry_status = "Started"

        self.config = None
        self.sensors = self.cfg.obs
        self._debug_mode = False
        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None
        timeout = float(self.cfg.arguments.timeout)
        watchdog_timeout = max(5, timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def _init_carla_world(self):
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # ### 1.set routes
        self.route_indexer = RouteIndexer(self.cfg.arguments.routes, self.cfg.arguments.scenarios, self.cfg.arguments.repetitions)

    def _load_and_wait_for_world(self, arguments, town):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.world = self.client.load_world(town)

        # print("self.world:",self.world)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(arguments.trafficManagerPort))
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(arguments.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))
    # 和_load_and_wait_for_world同时被调用
    def _prepare_ego_vehicles(self, config):
        """
        Spawn or update the ego vehicles
        """
        wait_for_ego_vehicles=False
        self.ego_vehicles=config.ego_vehicles
        if not wait_for_ego_vehicles:
            for vehicle in self.ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                                vehicle.transform,
                                                                                vehicle.rolename,
                                                                                color=vehicle.color,
                                                                                vehicle_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in self.ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break
            
            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(self.ego_vehicles[i].transform)
        # self.hero_vehicle = ego_vehicles[0]
        # sync state
        CarlaDataProvider.get_world().tick()

    def setup_route_scenario(self, world, config, arguments):
        route_scenario = RouteScenario(world=world, config=config, debug_mode=arguments.debug)
        self.statistics_manager.set_scenario(route_scenario.scenario)

        # Night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in route_scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        if arguments.record:
            self.client.start_recorder("{}/{}_rep{}.log".format(arguments.record, config.name, config.repetition_index))
    
        ### 5.Load scenario and run it
        #load_scenario(route_scenario, agent_instance, config.repetition_index)
        GameTime.restart()

        self.scenario_class = route_scenario
        self.scenario = route_scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = route_scenario.ego_vehicles
        self.other_actors = route_scenario.other_actors
        self.repetition_number = config.repetition_index

    # 需要配合sensor_helper使用
    def setup_sensors(self):
        self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
        # pass
        CarlaDataProvider.set_world(self.world)
        self._sensor_helper = SensorHelper(self.cfg.obs, None)
        self._sensor_helper.setup_sensors(self.world, self.ego_vehicles[0])

    def calculate_speed(self, actor):
        """
        Method to calculate the velocity of a actor
        """
        speed_squared = actor.get_velocity().x ** 2
        speed_squared += actor.get_velocity().y ** 2
        speed_squared += actor.get_velocity().z ** 2
        return math.sqrt(speed_squared)

    def step(self, action):
        try:
            obs=self.get_observation()
            # print("obs:",obs)
            reward={}
            done=False
            info={}
            self.action=action

            self.ego_vehicles[0].apply_control(self.action)      
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)
            
            if cfg.arguments.record:
                self.client.stop_recorder()

            self.close()
        return obs, reward, done, info
    
    def get_observation(self):
        sensor_data = self._sensor_helper.get_sensors_data()

        return sensor_data 

    def reset(self, cfg, config):
        # run起来
        # 返回obs
        self.config = config
        try:
            self.statistics_manager.set_route(config.name, config.index)
            if self.cfg.arguments.resume:
                self.route_indexer.resume(self.cfg.arguments.checkpoint)
                self.statistics_manager.resume(self.cfg.arguments.checkpoint)
            else:
                self.statistics_manager.clear_record(self.cfg.arguments.checkpoint)
                self.route_indexer.save_state(self.cfg.arguments.checkpoint)

            self._load_and_wait_for_world(cfg.arguments, config.town)
            self._prepare_ego_vehicles(config)
            self.setup_route_scenario(self.world, config, self.cfg.arguments)
            self.setup_sensors()
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)
            
            if cfg.arguments.record:
                self.client.stop_recorder()

            self.close()

        return self.get_observation()    
    
    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time
        # print("self.get_running_status():",self.get_running_status())
        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()
            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'
        # print("self.scenario.get_criteria():",self.scenario.get_criteria())
        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'
        # print("global_result:",global_result)
        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'
        # 数据都在global_result里面
        ResultOutputProvider(self, global_result)


    def close(self):
        try:
            if self.get_running_status() \
                    and hasattr(self, 'world') and self.world:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.traffic_manager.set_synchronous_mode(False)


            print("\033[1m> Stopping the route\033[0m")
            self.stop_scenario()
            self._register_statistics(self.config, self.cfg.arguments.checkpoint, self.entry_status, self.crash_message)
            
            if self.cfg.arguments.record:
                self.client.stop_recorder()
            CarlaDataProvider.cleanup()

            for i, _ in enumerate(self.ego_vehicles):
                if self.ego_vehicles[i]:
                    self.ego_vehicles[i].destroy()
                    self.ego_vehicles[i] = None
            self.ego_vehicles = []
            self._timestamp_last_run = 0.0
            self.scenario_duration_system = 0.0
            self.scenario_duration_game = 0.0
            self.start_system_time = None
            self.end_system_time = None
            self.end_game_time = None
            # print("=====closed!")

            if self.crash_message == "Simulation crashed":
                sys.exit(-1)

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"


    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.scenario_duration_system,
            self.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")

        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)


    def compute_reward(self):
        pass 



    def __repr__(self) -> str:
        return "ScenarioLeaderboardEnv with host %s, port %s." % (self._carla_host, self._carla_port)
    
    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()
    
    def tick_carla_world(self):
        try:
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            # print("timestamp:",timestamp)
            if timestamp:
                # run_eval._tick_scenario(timestamp,statistics_manager,config)
                if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
                    self._timestamp_last_run = timestamp.elapsed_seconds
                    self._watchdog.update()
                    GameTime.on_carla_tick(timestamp)
                    CarlaDataProvider.on_carla_tick()
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91m Terminated:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(self.config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)

        return timestamp
    
    def adjust_world_transform(self):
        if self._debug_mode:
            print("\n")
            py_trees.display.print_ascii_tree(
                self.scenario_tree, show_status=True)
            sys.stdout.flush()
        # print("self.scenario_tree.status：",self.scenario_tree.status)
        self.scenario_tree.tick_once()
        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False

        try:
        # 调整carla_world视角
            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))
                # if self._running and getself._running_status():
            if self._running and self.get_running_status():
                CarlaDataProvider.get_world().tick()
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91m Terminated:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(self.config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)



