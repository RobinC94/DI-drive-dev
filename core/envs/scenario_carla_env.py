import math
import sys
from typing import Any, Dict
import carla
from gym import spaces
import numpy as np
import os
from easydict import EasyDict
import time
import traceback
import py_trees

import carla
from core.utils.planner.scenario_planner import ScenarioPlanner
from core.utils.simulator_utils.srunner_utils import interpolate_trajectory_didrive, VoidAgent
from .base_drive_env import BaseDriveEnv
from core.utils.simulator_utils.sensor_helper import SensorHelper

from srunner.scenariomanager.traffic_events import TrafficEventType
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.result_writer import ResultOutputProvider
from leaderboard.scenarios.route_scenario import RouteScenario


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
    'sensor.speedometer':       'carla_speedometer',
    'sensor.other.collision':   'carla_collision'
}
PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80


def compute_route_length(config):
    trajectory = config.trajectory

    route_length = 0.0
    previous_location = None
    for location in trajectory:
        if previous_location:
            dist = math.sqrt(
                (location.x - previous_location.x) * (location.x - previous_location.x) +
                (location.y - previous_location.y) * (location.y - previous_location.y) +
                (location.z - previous_location.z) * (location.z - previous_location.z)
            )
            route_length += dist
        previous_location = location

    return route_length


class RouteRecord():

    def __init__(self):
        self.route_id = None
        self.index = None
        self.status = 'Started'
        self.infractions = {
            'collisions_pedestrian': [],
            'collisions_vehicle': [],
            'collisions_layout': [],
            'red_light': [],
            'stop_infraction': [],
            'outside_route_lanes': [],
            'route_dev': [],
            'route_timeout': [],
            'vehicle_blocked': []
        }

        self.scores = {'score_route': 0, 'score_penalty': 0, 'score_composed': 0}

        self.meta = {}


class ScenarioCarlaEnv(BaseDriveEnv):

    config = dict(
        obs=[],
        timeout=60.0,
        debug=False,
    )

    _planner_cfg = dict()

    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})
    reward_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))

    # Tunable parameters
    _client_timeout = 10.0  # in seconds
    _frame_rate = 20.0  # in Hz

    def __init__(
            self,
            cfg: Dict,
            host: str = 'localhost',
            port: int = 2000,
            tm_port: int = 2500,
            **kwargs,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self._statistics_manager = StatisticsManager()
        self._tm_port = tm_port
        self._client = carla.Client(host, port)

        self._client.set_timeout(self._client_timeout)
        self._traffic_manager = self._client.get_trafficmanager(int(self._tm_port))
        self._timestamp_last_run = 0.0
        self.ego_vehicles = []
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0

        self.config = None
        self.scenario = None
        self._sensors = self._cfg.obs
        self._start_time = GameTime.get_time()
        self._end_time = None
        timeout = float(self._cfg.timeout)
        watchdog_timeout = max(5, timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)
        self._collided = False
        self._tick = 0
        self._timestamp = 0

    def _load_planner(self):
        gps_route, route = interpolate_trajectory_didrive(self.world, self._config.trajectory)
        self._planner = ScenarioPlanner(self._planner_cfg, CarlaDataProvider)
        self._planner.set_route(route, clean=True)

    def _load_and_wait_for_world(self, town):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        self.start_system_time = time.time()
        self._start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.world = self._client.load_world(town)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self._frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self._client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(self._tm_port))
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_random_device_seed(int(self._tm_port))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!" "This scenario requires to use map {}".format(town))

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

    def _prepare_ego_vehicles(self, config):
        """
        Spawn or update the ego vehicles
        """
        wait_for_ego_vehicles = False
        self.ego_vehicles = config.ego_vehicles
        if not wait_for_ego_vehicles:
            for vehicle in self.ego_vehicles:
                self.ego_vehicles.append(
                    CarlaDataProvider.request_new_actor(
                        vehicle.model,
                        vehicle.transform,
                        vehicle.rolename,
                        color=vehicle.color,
                        vehicle_category=vehicle.category
                    )
                )
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
        CarlaDataProvider.get_world().tick()

    def _setup_route_scenario(self, world, config):
        route_scenario = RouteScenario(world=world, config=config, debug_mode=self._cfg.debug)
        self._statistics_manager.set_scenario(route_scenario.scenario)
        self.route_record = RouteRecord()

        # Night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in route_scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        # if arguments.record:
        #     self.client.start_recorder("{}/{}_rep{}.log".format(arguments.record, config.name, config.repetition_index))

        GameTime.restart()

        self._scenario_class = route_scenario
        self.scenario = route_scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = route_scenario.ego_vehicles
        self.other_actors = route_scenario.other_actors
        self.repetition_number = config.repetition_index

    def _setup_sensors(self):
        self._sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self._sensors]
        # pass
        CarlaDataProvider.set_world(self.world)
        self._sensor_helper = SensorHelper(self._sensors, None)
        self._sensor_helper.setup_sensors(self.world, self.ego_vehicles[0])
        # self._collision_sensor = CollisionSensor(self.ego_vehicles[0], 400)
        # self._traffic_light_helper = TrafficLightHelper(CarlaDataProvider, self.ego_vehicles[0])

    def _stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self._end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self._end_game_time - self._start_game_time
        # print("self._get_running_status():",self._get_running_status())
        if self._get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()
            self._analyze_scenario()

    def _analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'
        # print("self.scenario.get_criteria():",self.scenario.get_criteria())
        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'
        # print("global_result:",global_result)
        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'
        # 数据都在global_result里面
        ResultOutputProvider(self, global_result)

    def get_reward(self):
        self.compute_reward()
        # 这里的设计，应该是放到全局的list里，然后取最后一个，这样可以存下过程的reward
        score_route = self.route_record.scores['score_route']
        score_penalty = self.route_record.scores['score_penalty']
        score_composed = self.route_record.scores['score_composed']
        reward = {
            "score_composed": score_composed,
            "score_route": score_route,
            "score_penalty": score_penalty,
        }
        return reward

    def compute_reward(self, duration_time_system=-1, duration_time_game=-1, failure=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = self._config.index
        # if not self._registry_route_records or index >= len(self._registry_route_records):
        #     raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        # route_record = self._registry_route_records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        # meta['duration_system'] = duration_time_system
        # route_record.meta['duration_game'] = duration_time_game
        self.route_record.meta['route_length'] = compute_route_length(self._config)

        if self.scenario:
            if self.scenario.timeout_node.timeout:
                # route_record.infractions['route_timeout'].append('Route timeout.')
                failure = "Agent timed out"

            # 切割一下字符串
            for node in self.scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            # route_record.infractions['collisions_layout'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            # route_record.infractions['collisions_pedestrian'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            # route_record.infractions['collisions_vehicle'].append(event.get_message())
                            # print("event.get_message():",event.get_message())

                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                            # route_record.infractions['outside_route_lanes'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            # route_record.infractions['red_light'].append(event.get_message())

                        # elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        #     route_record.infractions['route_dev'].append(event.get_message())
                        #     failure = "Agent deviated from the route"

                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            # route_record.infractions['stop_infraction'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            # route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = "Agent got blocked"

                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0

        # update route scores
        self.route_record.scores['score_route'] = score_route
        self.route_record.scores['score_penalty'] = score_penalty
        self.route_record.scores['score_composed'] = max(score_route * score_penalty, 0.0)

        # update status
        if target_reached:
            self.route_record.status = 'Completed'
        else:
            self.route_record.status = 'Failed'
            if failure:
                self.route_record.status += ' - ' + failure

        # return route_record

    def _get_vehicle_control(self, actions):
        control = carla.VehicleControl()
        if actions:
            if 'steer' in actions:
                control.steer = float(actions['steer'])
            if 'throttle' in actions:
                control.throttle = float(actions['throttle'])
            if 'brake' in actions:
                control.brake = float(actions['brake'])
        return control

    def step(self, action=None):
        try:
            self._planner.run_step()
            timestamp = self._tick_carla_world()
            self._tick += 1
            self._timestamp = timestamp.elapsed_seconds
            # self._collided = self._collision_sensor.collided
            if timestamp and self._running:
                obs = self.get_observation()
                reward = self.get_reward()['score_composed']
                done = not self._running
                info = {}
                self.action = self._get_vehicle_control(action)

                self.ego_vehicles[0].apply_control(self.action)
                self._adjust_world_transform()
            else:
                raise ValueError
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            # self._register_statistics(config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)

            # if cfg.arguments.record:
            #     self.client.stop_recorder()

            self.close()
            raise e
        return obs, reward, done, info

    def get_information(self) -> Dict:
        """
        Get running information including time and ran light counts in current world.

        :Returns:
            Dict: Information dict.
        """
        information = {
            'tick': self._tick,
            'timestamp': self._timestamp,
            # 'total_lights': self._traffic_light_helper.total_lights,
            # 'total_lights_ran': self._traffic_light_helper.total_lights_ran
        }

        return information

    def get_observation(self):
        sensor_data = self._sensor_helper.get_sensors_data()

        for k, v in sensor_data.items():
            if v is None or len(v) == 0:
                return {}

        speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0]) * 3.6
        transform = CarlaDataProvider.get_transform(self.ego_vehicles[0])
        location = transform.location
        forward_vector = transform.get_forward_vector()
        drive_waypoint = CarlaDataProvider._map.get_waypoint(
            location,
            project_to_road=False,
        )
        is_junction = False
        if drive_waypoint is not None:
            is_junction = drive_waypoint.is_junction
            self._off_road = False
        else:
            self._off_road = True
        lane_waypoint = CarlaDataProvider._map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        lane_location = lane_waypoint.transform.location
        lane_forward_vector = lane_waypoint.transform.rotation.get_forward_vector()

        state = {
            'speed': speed,
            'location': np.array([location.x, location.y, location.z]),
            'forward_vector': np.array([forward_vector.x, forward_vector.y]),
            # 'acceleration': np.array([acceleration.x, acceleration.y, acceleration.z]),
            # 'velocity': np.array([velocity.x, velocity.y, velocity.z]),
            # 'angular_velocity': np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z]),
            'rotation': np.array([transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]),
            'is_junction': is_junction,
            'lane_location': np.array([lane_location.x, lane_location.y]),
            'lane_forward': np.array([lane_forward_vector.x, lane_forward_vector.y]),
            # 'tl_state': light_state,
            # 'tl_dis': self._traffic_light_helper.active_light_dis,
        }

        if lane_waypoint is None:
            state['lane_forward'] = None
        else:
            lane_forward_vector = lane_waypoint.transform.get_forward_vector()
            state['lane_forward'] = np.array([lane_forward_vector.x, lane_forward_vector.y])

        navigation = self.get_navigation()
        information = self.get_information()

        obs = {}
        obs.update(sensor_data)
        obs.update(navigation)
        obs.update(state)
        obs.update(information)

        return obs

    def get_navigation(self):
        navigation = {}
        if not self._planner:
            return navigation
        command = self._planner.node_road_option
        node_location = self._planner.node_waypoint.transform.location
        node_forward = self._planner.node_waypoint.transform.rotation.get_forward_vector()
        target_location = self._planner.target_waypoint.transform.location
        target_forward = self._planner.target_waypoint.transform.rotation.get_forward_vector()
        waypoint_list = self._planner.get_waypoints_list(20)
        direction_list = self._planner.get_direction_list(20)
        agent_state = self._planner.agent_state
        speed_limit = self._planner.speed_limit
        self._end_distance = self._planner.distance_to_goal
        self._end_timeout = self._planner.timeout

        waypoint_location_list = []
        for wp in waypoint_list:
            wp_loc = wp.transform.location
            wp_vec = wp.transform.rotation.get_forward_vector()
            waypoint_location_list.append([wp_loc.x, wp_loc.y, wp_vec.x, wp_vec.y])

        if not self._off_road:
            current_waypoint = self._planner.current_waypoint
            node_waypoint = self._planner.node_waypoint

            # Lanes and roads are too chaotic at junctions
            if current_waypoint.is_junction or node_waypoint.is_junction:
                self._wrong_direction = False
            else:
                node_yaw = node_waypoint.transform.rotation.yaw % 360
                cur_yaw = current_waypoint.transform.rotation.yaw % 360

                wp_angle = (node_yaw - cur_yaw) % 360

                if 150 <= wp_angle <= (360 - 150):
                    self._wrong_direction = True
                else:
                    # Changing to a lane with the same direction
                    self._wrong_direction = False

        navigation = {
            'agent_state': agent_state.value,
            'command': command.value,
            'node': np.array([node_location.x, node_location.y]),
            'node_forward': np.array([node_forward.x, node_forward.y]),
            'target': np.array([target_location.x, target_location.y]),
            'target_forward': np.array([target_forward.x, target_forward.y]),
            'waypoint_list': np.array(waypoint_location_list),
            'speed_limit': np.array(speed_limit),
            'direction_list': np.array(direction_list)
        }

        return navigation

    def reset(self, config) -> Any:
        self._config = config
        self._config.agent = VoidAgent()
        try:
            while True:
                self._statistics_manager.set_route(config.name, config.index)

                self._load_and_wait_for_world(config.town)
                self._prepare_ego_vehicles(config)
                self._setup_route_scenario(self.world, config)
                self._load_planner()
                self._setup_sensors()
                if self._ready():
                    break
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self.close()
        return self.get_observation()

    def close(self):
        try:
            if self._get_running_status() \
                    and hasattr(self, 'world') and self.world:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self._traffic_manager.set_synchronous_mode(False)

            print("\033[1m> Stopping the route\033[0m")
            self._stop_scenario()

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
            self._end_game_time = None
            self._collided = False
            # print("=====closed!")

            if self.crash_message == "Simulation crashed":
                sys.exit(-1)

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def _tick_carla_world(self):
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
                    #CarlaDataProviderExpand.on_carla_tick()
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91m Terminated:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            #self._register_statistics(self.config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)

        return timestamp

    def _adjust_world_transform(self):
        if self._cfg.debug:
            print("\n")
            py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
            sys.stdout.flush()
        # print("self.scenario_tree.status：",self.scenario_tree.status)
        self.scenario_tree.tick_once()
        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False

        try:
            # 调整carla_world视角
            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(
                carla.Transform(ego_trans.location + carla.Location(z=50), carla.Rotation(pitch=-90))
            )
            # if self._running and getself._running_status():
            if self._running and self._get_running_status():
                CarlaDataProvider.get_world().tick()
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91m Terminated:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            #self._register_statistics(self.config, cfg.arguments.checkpoint, self.entry_status, self.crash_message)

    def _ready(self, ticks: int = 30) -> bool:
        for _ in range(ticks):
            self.step()
            self.get_observation()
        # print("ready!!!!!!!")
        self._tick = 0
        self._timestamp = 0

        return not self._collided

    @property
    def collided(self) -> bool:
        return self._collided

    @property
    def running(self) -> bool:
        return self._running
