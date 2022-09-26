import math
import numpy as np
from enum import Enum
from collections import deque
from easydict import EasyDict
from typing import Dict, List, Tuple, Union
import copy
import carla
import shapely
from core.utils.env_utils.carla_data_provider_expand import CarlaDataProviderExpand
from core.utils.simulator_utils.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from core.utils.simulator_utils.agents.navigation.global_route_planner import GlobalRoutePlanner
from core.utils.simulator_utils.agents.navigation import RoadOption
from core.utils.simulator_utils.agents.tools.misc import draw_waypoints, is_within_distance_ahead
from ding.utils.default_helper import deep_merge_dicts


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    VOID = -1
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_BY_WALKER = 3
    BLOCKED_RED_LIGHT = 4
    BLOCKED_BY_BIKE = 5


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


class AutoPIDPlanner(object):
    """
    Basic local planner for Carla simulator. It can set route with a pair of start and end waypoints,
    or directly set with a waypoint list. The planner will provide target waypoint and road option
    in current route position, and record current route distance and end timeout. The planner will
    also judge agent state by checking surrounding vehicles, walkers and traffic lights.

    The route's element consists of a waypoint and a road option. Local planner uses a waypoint queue
    to store all the unreached waypoints, and a waypoint buffer to store some of the near waypoints to
    speed up searching. In short, `node` waypoint is the waypoint in route that farthest from hero
    vehicle and within ``min_distance``, and `target` waypoint is the next waypoint of node waypoint.

    :Arguments:
        - cfg (Dict): Config dict.

    :Interfaces: set_destination, set_route, run_step, get_waypoints_list, clean_up
    """

    config = dict(
        min_distance=5.0,
        resolution=5.0,
        fps=10,
        debug=False,
    )

    def __init__(self, cfg: Dict, carla_interface: 'CarlaInterface') -> None:  # noqa
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self._carla_interface = carla_interface
        self._hero_vehicle = self._carla_interface.get_hero_actor()
        self._world = self._carla_interface._world
        self._map = self._carla_interface.get_map()

        self._resolution = self._cfg.resolution
        self._min_distance = self._cfg.min_distance
        self._fps = self._cfg.fps

        self._route = []
        self._waypoints_queue = deque()
        self._buffer_size = 100
        self._waypoints_buffer = deque(maxlen=100)
        self._end_location = None

        self.current_waypoint = None
        self.node_waypoint = None
        self.target_waypoint = None
        self.node_road_option = None
        self.target_road_option = None
        self.agent_state = None
        self.speed_limit = 0

        self.distance_to_goal = 0.0
        self.distances = deque()
        self.timeout = -1
        self.timeout_in_seconds = 0

        self._debug = self._cfg.debug

        self._actor_speed_vector_map = dict()

        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, self._resolution))
        self._grp.setup()

    def set_destination(self, start_location, end_location, clean: bool = False) -> None:
        """
        This method creates a route of a list of waypoints from start location to destination location
        based on the route traced by the global router. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - start_location (carla.Location): initial position.
            - end_location (carla.Location): final position.
            - clean (bool): Whether to clean current route. Defaults to False.
        """
        start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)
        new_route = self._grp.trace_route(start_waypoint.transform.location, self.end_waypoint.transform.location)
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = new_route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route += new_route
        # print("self._route:", self._route)
        self._carla_interface.set_ego_vehicle_route(self._route)

        prev_loc = None
        for elem in new_route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        self._buffer_size = min(int(100 // self._resolution), 100)
        self.node_waypoint = start_waypoint
        self.node_road_option = RoadOption.LANEFOLLOW
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def set_route(self, route: List, clean: bool = False) -> None:
        """
        This method add a route into planner to trace. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - route (List): Route add to planner.
            - clean (bool, optional): Whether to clean current route. Defaults to False.
        """
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route.extend(route)

        self.end_waypoint = self._route[-1][0]

        # self._carla_interface.set_hero_vehicle_route(self._route)
        prev_loc = None
        for elem in route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        if self.distances:
            cur_resolution = np.average(list(self.distances)[:100])
            self._buffer_size = min(100, int(100 // cur_resolution))
        self.node_waypoint, self.node_road_option = self._waypoints_queue[0]
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def add_route_in_front(self, route):
        if self._waypoints_buffer:
            prev_loc = self._waypoints_buffer[0][0].transform.location
        else:
            prev_loc = self._waypoints_queue[0][0].transform.location
        for elem in route[::-1]:
            self._waypoints_buffer.appendleft(elem)
            cur_loc = elem[0].transform.location
            delta = cur_loc.distance(prev_loc)
            self.distance_to_goal += delta
            self.distances.appendleft(delta)
            prev_loc = cur_loc

        if len(self._waypoints_buffer) > self._buffer_size:
            for i in range(len(self._waypoints_buffer) - self._buffer_size):
                elem = self._waypoints_buffer.pop()
                self._waypoints_queue.appendleft(elem)
        self.node_waypoint, self.node_road_option = self._waypoints_buffer[0]

    def run_step(self) -> None:
        """
        Run one step of local planner. It will update node and target waypoint and road option, and check agent
        states.
        """
        assert self._route is not None

        vehicle_transform = self._carla_interface.get_transform(self._hero_vehicle)
        if not vehicle_transform:
            return
        self.current_waypoint = self._map.get_waypoint(
            vehicle_transform.location, lane_type=carla.LaneType.Driving, project_to_road=True
        )

        # print("self.current_waypoint:", self.current_waypoint)
        # 问题就出在这个函数，没有更新，怎么让它更新呢？尝试解决一下吧
        # Add waypoints into buffer if empty
        if not self._waypoints_buffer:
            for i in range(min(self._buffer_size, len(self._waypoints_queue))):
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                else:
                    break

            # If no waypoints return with current waypoint
            if not self._waypoints_buffer:
                self.target_waypoint = self.current_waypoint
                self.node_waypoint = self.current_waypoint
                self.target_road_option = RoadOption.VOID
                self.node_road_option = RoadOption.VOID
                self.agent_state = AgentState.VOID
                return
        # print("self._waypoints_buffer:", self._waypoints_buffer)
        # Find the most far waypoint within min distance
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoints_buffer):
            # print("waypoint:", waypoint)
            cur_dis = waypoint.transform.location.distance(vehicle_transform.location)
            # print("cur_dis:", cur_dis)
            if cur_dis < self._min_distance:
                # print("cur_dis:", cur_dis)
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self.node_waypoint, self.node_road_option = self._waypoints_buffer.popleft()
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                if self.distances:
                    self.distance_to_goal -= self.distances.popleft()
        # Update information
        if self._waypoints_buffer:
            self.target_waypoint, self.target_road_option = self._waypoints_buffer[0]
            # print("self.target_waypoint:", self.target_waypoint)
        self.speed_limit = self._hero_vehicle.get_speed_limit()
        self.agent_state = AgentState.NAVIGATING

        # Detect vehicle and light hazard
        # 这块需要改造，改造完就成了
        vehicle_state, vehicle = self.is_vehicle_hazard(self._carla_interface, self._hero_vehicle)
        # print("vehicle_state:", vehicle_state)
        # print("vehicle:", vehicle)
        if not vehicle_state:
            vehicle_state, vehicle = self.is_lane_vehicle_hazard(self._hero_vehicle, self.target_road_option)
            # print("vehicle: ",vehicle,"lane_vehicle_hazard vehicle_state:", vehicle_state)

        if not vehicle_state:
            # print("is_junction_vehicle_hazard")
            vehicle_state, vehicle = self.is_junction_vehicle_hazard(self._hero_vehicle, self.target_road_option)
            # print("vehicle: ",vehicle,",junction_vehicle_hazard vehicle_state:", vehicle_state)

        if vehicle_state:
            if self._debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))
            # print("AgentState.BLOCKED_BY_VEHICLE")
            self.agent_state = AgentState.BLOCKED_BY_VEHICLE

        bike_state, bike = self.is_bike_hazard(self._hero_vehicle)
        if bike_state:
            if self._debug:
                print('!!! BIKE BLOCKING AHEAD [{}])'.format(bike.id))

            self.agent_state = AgentState.BLOCKED_BY_BIKE

        walker_state, walker = self.is_walker_hazard(self._hero_vehicle)
        if walker_state:
            if self._debug:
                print('!!! WALKER BLOCKING AHEAD [{}])'.format(walker.id))

            self.agent_state = AgentState.BLOCKED_BY_WALKER

        # light_state, traffic_light = self.is_light_red(self._hero_vehicle)

        # if light_state:
        #     if self._debug:
        #         print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

        #     self.agent_state = AgentState.BLOCKED_RED_LIGHT
        # print("self.current_waypoint:", self.current_waypoint)
        if self._debug:
            draw_waypoints(self._world, self.current_waypoint)

    def get_speed(self, actor: carla.Actor) -> float:
        """
        returns the absolute speed for the given actor
        """
        for key in self._carla_interface._actor_velocity_map:
            if key.id == actor.id:
                # print("speed:", self._carla_interface._actor_velocity_map[key])
                return self._carla_interface._actor_velocity_map[key]
        print('WARNING: {}.get_speed: {} not found!'.format(__name__, actor))
        return -1

    def is_light_red(self, vehicle: carla.Actor, proximity_tlight_threshold: float = 10.0):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.
        :Arguments:
            - proximity_tlight_threshold: threshold to judge if light affecting
        :Returns: a tuple given by (bool_flag, traffic_light), where
            - bool_flag: True if there is a traffic light in RED affecting us and False otherwise
            - traffic_light: The object itself or None if there is no red traffic light affecting us
        """
        lights_list = self.get_actor_list().filter("*traffic_light*")

        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        for traffic_light in lights_list:
            object_location = self.get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            if is_within_distance_ahead(object_waypoint.transform, vehicle.get_transform(), proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def get_trafficlight_trigger_location(self, traffic_light: carla.Actor) -> carla.Vector3D:  # pylint: disable=invalid-name
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point, angle):
            """
            rotate a given point by a given angle
            """
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def is_walker_hazard(self, vehicle: carla.Actor):
        """
        :Arguments:
            - vehicle: Potential obstacle to check
        :Returns: a tuple given by (bool_flag, vehicle), where
            - bool_flag: True if there is a walker ahead blocking us and False otherwise
            - walker: The blocker object itself
        """
        walkers_list = self.get_actor_list().filter("*walker.*")
        p1 = _numpy(self.get_location(vehicle))
        v1 = 10.0 * _orientation(self._carla_interface.get_transform(vehicle).rotation.yaw)

        for walker in walkers_list:
            if not isinstance(walker, carla.Walker):
                continue
            v2_hat = _orientation(self._carla_interface.get_transform(walker).rotation.yaw)
            s2 = self.get_speed(walker)

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(self.get_location(walker))
            v2 = 8.0 * v2_hat

            collides, collision_point = self.get_collision(p1, v1, p2, v2)

            if collides:
                return (True, walker)
        return (False, None)

    def is_bike_hazard(self, vehicle):
        bikes_list = self.get_actor_list().filter("*vehicle*")
        o1 = _orientation(self._carla_interface.get_transform(vehicle).rotation.yaw)
        v1_hat = o1
        p1 = _numpy(self.get_location(vehicle))
        v1 = 10.0 * o1

        for bike in bikes_list:
            if 'driver_id' not in bike.attributes or not bike.is_alive:
                continue
            o2 = _orientation(self._carla_interface.get_transform(bike).rotation.yaw)
            s2 = self.get_speed(bike)
            v2_hat = o2
            p2 = _numpy(self.get_location(bike))

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            if distance > 20:
                continue
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
            if angle_to_car > 30:
                continue
            if angle_between_heading < 75 and angle_between_heading > 105:
                continue

            p2_hat = -2.0 * v2_hat + p1
            v2 = 8.0 * v2_hat

            collides, collision_point = self.get_collision(p1, v1, p2_hat, v2)

            if collides:
                return (True, bike)

        return (False, None)

    def is_junction_vehicle_hazard(self, vehicle: carla.Actor, command: RoadOption):
        """
        :Arguments:
            - vehicle: Potential obstacle to check
            - command: future driving command
        :Returns: a tuple given by (bool_flag, vehicle), where
            - bool_flag: True if there is a vehicle ahead blocking us in junction and False otherwise
            - vehicle: The blocker object itself
        """
        vehicle_list = self.get_actor_list().filter("*vehicle*")
        o1 = _orientation(self._carla_interface.get_transform(vehicle).rotation.yaw)
        x1 = vehicle.bounding_box.extent.x
        p1 = self.get_location(vehicle) + self._carla_interface.get_transform(vehicle).get_forward_vector()
        w1 = self._map.get_waypoint(p1)
        s1 = self.get_speed(vehicle)
        if command == RoadOption.RIGHT:
            shift_angle = 25
        elif command == RoadOption.LEFT:
            shift_angle = -25
        else:
            shift_angle = 0
        v1 = (5 * s1 + 6) * _orientation(self._carla_interface.get_transform(vehicle).rotation.yaw + shift_angle)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == vehicle.id or not target_vehicle.is_alive:
                continue

            o2 = _orientation(self._carla_interface.get_transform(target_vehicle).rotation.yaw)
            o2_left = _orientation(self._carla_interface.get_transform(target_vehicle).rotation.yaw - 15)
            o2_right = _orientation(self._carla_interface.get_transform(target_vehicle).rotation.yaw + 15)
            x2 = target_vehicle.bounding_box.extent.x

            p2 = self.get_location(target_vehicle)
            p2_hat = p2 - (x2 + 2) * self._carla_interface.get_transform(target_vehicle).get_forward_vector()
            w2 = self._map.get_waypoint(p2)
            s2 = self.get_speed(target_vehicle)

            v2 = (4 * s2 + 2 * x2 + 7) * o2
            v2_left = (4 * s2 + 2 * x2 + 7) * o2_left
            v2_right = (4 * s2 + 2 * x2 + 7) * o2_right

            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            if vehicle.get_location().distance(p2) > 25:
                continue
            if w1.is_junction is False and w2.is_junction is False:
                continue
            if (angle_between_heading < 15.0 or angle_between_heading > 165) and command == RoadOption.STRAIGHT:
                continue
            collides, collision_point = self.get_collision(_numpy(p1), v1, _numpy(p2_hat), v2)
            if collides is None:
                collides, collision_point = self.get_collision(_numpy(p1), v1, _numpy(p2_hat), v2_left)
            if collides is None:
                collides, collision_point = self.get_collision(_numpy(p1), v1, _numpy(p2_hat), v2_right)
                continue
            if collides:
                return (True, target_vehicle)
        return (False, None)

    def get_collision(self, p1, v1, p2, v2):
        A = np.stack([v1, -v2], 1)
        b = p2 - p1

        if abs(np.linalg.det(A)) < 1e-3:
            return False, None

        x = np.linalg.solve(A, b)
        collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

        return collides, p1 + x[0] * v1

    def is_vehicle_hazard(self, _carla_interface, vehicle: carla.Actor, proximity_vehicle_threshold: float = 10.0):
        vehicle_list = self.get_actor_list().filter("*vehicle*")
        vehicle_location = self.get_location(vehicle)
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle.id or not target_vehicle.is_alive:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(), self._carla_interface.get_transform(vehicle),
                                        proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)

    def is_lane_vehicle_hazard(self, vehicle: carla.Actor, command: RoadOption):
        """
        :Arguments:
            - vehicle: Potential obstacle to check
            - command: future driving command
        :Returns: a tuple given by (bool_flag, vehicle), where
            - bool_flag: True if there is a vehicle in other lanes blocking us and False otherwise
            - vehicle: The blocker object itself
        """
        vehicle_list = self.get_actor_list().filter("*vehicle*")
        if command != RoadOption.CHANGELANELEFT and command != RoadOption.CHANGELANERIGHT:
            # print("command:", command)
            return (False, None)
        w1 = self._map.get_waypoint(vehicle.get_location())
        o1 = _orientation(self._carla_interface.get_transform(vehicle).rotation.yaw)
        p1 = self.get_location(vehicle)

        yaw_w1 = w1.transform.rotation.yaw
        lane_width = w1.lane_width
        location_w1 = w1.transform.location

        lft_shift = 0.5
        rgt_shift = 0.5
        if command == RoadOption.CHANGELANELEFT:
            rgt_shift += 1
        else:
            lft_shift += 1

        lft_lane_wp = self.rotate_point(carla.Vector3D(lft_shift * lane_width, 0.0, location_w1.z), yaw_w1 + 90)
        lft_lane_wp = location_w1 + carla.Location(lft_lane_wp)
        rgt_lane_wp = self.rotate_point(carla.Vector3D(rgt_shift * lane_width, 0.0, location_w1.z), yaw_w1 - 90)
        rgt_lane_wp = location_w1 + carla.Location(rgt_lane_wp)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == vehicle.id or not target_vehicle.is_alive:
                continue

            w2 = self._map.get_waypoint(self.get_location(target_vehicle))
            o2 = _orientation(self._carla_interface.get_transform(target_vehicle).rotation.yaw)
            p2 = self.get_location(target_vehicle)
            x2 = target_vehicle.bounding_box.extent.x
            p2_hat = p2 - self._carla_interface.get_transform(target_vehicle).get_forward_vector() * x2 * 2
            s2 = CarlaDataProviderExpand.get_speed_vector(
                target_vehicle
            ) + self._carla_interface.get_transform(target_vehicle).get_forward_vector() * x2
            s2_value = max(12, 2 + 2 * x2 + 3.0 * self.get_speed(target_vehicle))

            distance = p1.distance(p2)

            if distance > s2_value:
                continue
            if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
                continue
            if command == RoadOption.CHANGELANELEFT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
            if command == RoadOption.CHANGELANERIGHT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue

            if self.is_vehicle_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
                return (True, target_vehicle)
        return (False, None)

    def is_vehicle_crossing_future(self, p1, s1, lft_lane, rgt_lane):
        p1_hat = carla.Location(x=p1.x + 3 * s1.x, y=p1.y + 3 * s1.y)
        line1 = shapely.geometry.LineString([(p1.x, p1.y), (p1_hat.x, p1_hat.y)])
        line2 = shapely.geometry.LineString([(lft_lane.x, lft_lane.y), (rgt_lane.x, rgt_lane.y)])
        inter = line1.intersection(line2)
        return not inter.is_empty

    # def get_speed_vector(self, actor: carla.Actor):
    #     """
    #     returns the absolute speed for the given actor
    #     """
    #     # 还未初始化
    #     for key in self._actor_speed_vector_map:
    #         if key.id == actor.id:
    #             return self._actor_speed_vector_map[key]
    #     print('WARNING: {}.get_speed: {} not found!'.format(__name__, actor))
    #     return None

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_actor_list(self) -> List:
        """
        Return all actors in world
        """
        return self._world.get_actors()

    def get_location(self, actor: carla.Actor):
        """
        returns the location for the given actor
        """
        for key in self._carla_interface._actor_transform_map:
            if key.id == actor.id:
                return self._carla_interface._actor_transform_map[key].location
        print('WARNING: {}.get_location: {} not found!'.format(__name__, actor))
        return None

    def get_waypoints_list(self, waypoint_num: int) -> List[carla.Waypoint]:
        """
        Return a list of wapoints from the end of waypoint buffer.

        :Arguments:
            - waypoint_num (int): Num of waypoint in list.

        :Returns:
            List[carla.Waypoint]: List of waypoint.
        """
        num = 0
        i = 0
        waypoint_list = []
        while num < waypoint_num and i < len(self._waypoints_buffer):
            waypoint = self._waypoints_buffer[i][0]
            i += 1
            if len(waypoint_list) == 0:
                waypoint_list.append(waypoint)
                num + 1
            elif waypoint_list[-1].transform.location.distance(waypoint.transform.location) > 1e-4:
                waypoint_list.append(waypoint)
                num += 1
        return waypoint_list

    def get_direction_list(self, waypoint_num: int) -> List[RoadOption]:
        num = min(waypoint_num, len(self._waypoints_buffer))
        direction_list = []
        for i in range(num):
            direction = self._waypoints_buffer[i][1].value
            direction_list.append(direction)
        return direction_list

    def get_incoming_waypoint_and_direction(self, steps: int = 3) -> Tuple[carla.Waypoint, RoadOption]:
        """
        Returns direction and waypoint at a distance ahead defined by the user.

        :Arguments:
            - steps (int): Number of steps to get the incoming waypoint.

        :Returns:
            Tuple[carla.Waypoint, RoadOption]: Waypoint and road option ahead.
        """
        if len(self._waypoints_buffer) > steps:
            return self._waypoints_buffer[steps]
        elif (self._waypoints_buffer):
            return self._waypoints_buffer[-1]
        else:
            return self.current_waypoint, RoadOption.VOID

    def clean_up(self) -> None:
        """
        Clear route, waypoint queue and buffer.
        """
        self._waypoints_queue.clear()
        self._waypoints_buffer.clear()
        if self._route is not None:
            self._route.clear()
        self.distances.clear()

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)


class BasicPlanner(object):
    """
    Basic local planner for Carla simulator. It can set route with a pair of start and end waypoints,
    or directly set with a waypoint list. The planner will provide target waypoint and road option
    in current route position, and record current route distance and end timeout. The planner will
    also judge agent state by checking surrounding vehicles, walkers and traffic lights.

    The route's element consists of a waypoint and a road option. Local planner uses a waypoint queue
    to store all the unreached waypoints, and a waypoint buffer to store some of the near waypoints to
    speed up searching. In short, `node` waypoint is the waypoint in route that farthest from hero
    vehicle and within ``min_distance``, and `target` waypoint is the next waypoint of node waypoint.

    :Arguments:
        - cfg (Dict): Config dict.

    :Interfaces: set_destination, set_route, run_step, get_waypoints_list, clean_up
    """

    config = dict(
        min_distance=5.0,
        resolution=5.0,
        fps=10,
        debug=False,
    )

    def __init__(self, cfg: Dict) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self._hero_vehicle = CarlaDataProvider.get_hero_actor()
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()

        self._resolution = self._cfg.resolution
        self._min_distance = self._cfg.min_distance
        self._fps = self._cfg.fps

        self._route = None
        self._waypoints_queue = deque()
        self._buffer_size = 100
        self._waypoints_buffer = deque(maxlen=100)
        self._end_location = None

        self.current_waypoint = None
        self.node_waypoint = None
        self.target_waypoint = None
        self.node_road_option = None
        self.target_road_option = None
        self.agent_state = None
        self.speed_limit = 0

        self.distance_to_goal = 0.0
        self.distances = deque()
        self.timeout = -1
        self.timeout_in_seconds = 0

        self._debug = self._cfg.debug

        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, self._resolution))
        self._grp.setup()

    def set_destination(
            self, start_location: carla.Location, end_location: carla.Location, clean: bool = False
    ) -> None:
        """
        This method creates a route of a list of waypoints from start location to destination location
        based on the route traced by the global router. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - start_location (carla.Location): initial position.
            - end_location (carla.Location): final position.
            - clean (bool): Whether to clean current route. Defaults to False.
        """
        start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)
        new_route = self._grp.trace_route(start_waypoint.transform.location, self.end_waypoint.transform.location)
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = new_route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route += new_route
        CarlaDataProvider.set_hero_vehicle_route(self._route)

        prev_loc = None
        for elem in new_route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        self._buffer_size = min(int(100 // self._resolution), 100)
        self.node_waypoint = start_waypoint
        self.node_road_option = RoadOption.LANEFOLLOW
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def set_route(self, route: List, clean: bool = False) -> None:
        """
        This method add a route into planner to trace. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - route (List): Route add to planner.
            - clean (bool, optional): Whether to clean current route. Defaults to False.
        """
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route.extend(route)

        self.end_waypoint = self._route[-1][0]

        CarlaDataProvider.set_hero_vehicle_route(self._route)

        prev_loc = None
        for elem in route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        if self.distances:
            cur_resolution = np.average(list(self.distances)[:100])
            self._buffer_size = min(100, int(100 // cur_resolution))
        self.node_waypoint, self.node_road_option = self._waypoints_queue[0]
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def add_route_in_front(self, route):
        if self._waypoints_buffer:
            prev_loc = self._waypoints_buffer[0][0].transform.location
        else:
            prev_loc = self._waypoints_queue[0][0].transform.location
        for elem in route[::-1]:
            self._waypoints_buffer.appendleft(elem)
            cur_loc = elem[0].transform.location
            delta = cur_loc.distance(prev_loc)
            self.distance_to_goal += delta
            self.distances.appendleft(delta)
            prev_loc = cur_loc

        if len(self._waypoints_buffer) > self._buffer_size:
            for i in range(len(self._waypoints_buffer) - self._buffer_size):
                elem = self._waypoints_buffer.pop()
                self._waypoints_queue.appendleft(elem)
        self.node_waypoint, self.node_road_option = self._waypoints_buffer[0]

    def run_step(self) -> None:
        """
        Run one step of local planner. It will update node and target waypoint and road option, and check agent
        states.
        """
        assert self._route is not None

        vehicle_transform = CarlaDataProvider.get_transform(self._hero_vehicle)
        self.current_waypoint = self._map.get_waypoint(
            vehicle_transform.location, lane_type=carla.LaneType.Driving, project_to_road=True
        )

        # Add waypoints into buffer if empty
        if not self._waypoints_buffer:
            for i in range(min(self._buffer_size, len(self._waypoints_queue))):
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                else:
                    break

            # If no waypoints return with current waypoint
            if not self._waypoints_buffer:
                self.target_waypoint = self.current_waypoint
                self.node_waypoint = self.current_waypoint
                self.target_road_option = RoadOption.VOID
                self.node_road_option = RoadOption.VOID
                self.agent_state = AgentState.VOID
                return

        # Find the most far waypoint within min distance
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoints_buffer):
            cur_dis = waypoint.transform.location.distance(vehicle_transform.location)
            if cur_dis < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self.node_waypoint, self.node_road_option = self._waypoints_buffer.popleft()
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                if self.distances:
                    self.distance_to_goal -= self.distances.popleft()

        # Update information
        if self._waypoints_buffer:
            self.target_waypoint, self.target_road_option = self._waypoints_buffer[0]
        self.speed_limit = self._hero_vehicle.get_speed_limit()
        self.agent_state = AgentState.NAVIGATING

        # Detect vehicle and light hazard
        vehicle_state, vehicle = CarlaDataProvider.is_vehicle_hazard(self._hero_vehicle)
        if not vehicle_state:
            vehicle_state, vehicle = CarlaDataProvider.is_lane_vehicle_hazard(
                self._hero_vehicle, self.target_road_option
            )
        if not vehicle_state:
            vehicle_state, vehicle = CarlaDataProvider.is_junction_vehicle_hazard(
                self._hero_vehicle, self.target_road_option
            )
        if vehicle_state:
            if self._debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self.agent_state = AgentState.BLOCKED_BY_VEHICLE

        bike_state, bike = CarlaDataProvider.is_bike_hazard(self._hero_vehicle)
        if bike_state:
            if self._debug:
                print('!!! BIKE BLOCKING AHEAD [{}])'.format(bike.id))

            self.agent_state = AgentState.BLOCKED_BY_BIKE

        walker_state, walker = CarlaDataProvider.is_walker_hazard(self._hero_vehicle)
        if walker_state:
            if self._debug:
                print('!!! WALKER BLOCKING AHEAD [{}])'.format(walker.id))

            self.agent_state = AgentState.BLOCKED_BY_WALKER

        light_state, traffic_light = CarlaDataProvider.is_light_red(self._hero_vehicle)

        if light_state:
            if self._debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self.agent_state = AgentState.BLOCKED_RED_LIGHT

        if self._debug:
            draw_waypoints(self._world, self.current_waypoint)

    def get_waypoints_list(self, waypoint_num: int) -> List[carla.Waypoint]:
        """
        Return a list of wapoints from the end of waypoint buffer.

        :Arguments:
            - waypoint_num (int): Num of waypoint in list.

        :Returns:
            List[carla.Waypoint]: List of waypoint.
        """
        num = 0
        i = 0
        waypoint_list = []
        while num < waypoint_num and i < len(self._waypoints_buffer):
            waypoint = self._waypoints_buffer[i][0]
            i += 1
            if len(waypoint_list) == 0:
                waypoint_list.append(waypoint)
                num + 1
            elif waypoint_list[-1].transform.location.distance(waypoint.transform.location) > 1e-4:
                waypoint_list.append(waypoint)
                num += 1
        return waypoint_list

    def get_direction_list(self, waypoint_num: int) -> List[RoadOption]:
        num = min(waypoint_num, len(self._waypoints_buffer))
        direction_list = []
        for i in range(num):
            direction = self._waypoints_buffer[i][1].value
            direction_list.append(direction)
        return direction_list

    def get_incoming_waypoint_and_direction(self, steps: int = 3) -> Tuple[carla.Waypoint, RoadOption]:
        """
        Returns direction and waypoint at a distance ahead defined by the user.

        :Arguments:
            - steps (int): Number of steps to get the incoming waypoint.

        :Returns:
            Tuple[carla.Waypoint, RoadOption]: Waypoint and road option ahead.
        """
        if len(self._waypoints_buffer) > steps:
            return self._waypoints_buffer[steps]
        elif (self._waypoints_buffer):
            return self._waypoints_buffer[-1]
        else:
            return self.current_waypoint, RoadOption.VOID

    def clean_up(self) -> None:
        """
        Clear route, waypoint queue and buffer.
        """
        self._waypoints_queue.clear()
        self._waypoints_buffer.clear()
        if self._route is not None:
            self._route.clear()
        self.distances.clear()

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)
