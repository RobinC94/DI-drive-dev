import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

import sys
sys.path.insert(0,os.path.abspath('./leaderboard_package'))
sys.path.insert(0,os.path.abspath('./scenario_runner'))
sys.path.insert(0,os.path.abspath('./core/utils/simulator_utils'))

import carla
from core.envs import ScenarioCarlaEnv
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed, deep_merge_dicts
from core.utils.others.image_helper import show_image

from leaderboard.utils.route_indexer import RouteIndexer


casezoo_config = dict(
    env=dict(
        scenario=dict(
            repetitions=1,
        ),
        obs=[
        {
            "type": "sensor.camera.rgb",
            "x": 1.3,
            "y": 0.0,
            "z": 2.3,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "width": 800,
            "height": 600,
            "fov": 100,
            "id": "rgb",
        },
        {
            "type": "sensor.camera.rgb",
            "x": 1.3,
            "y": 0.0,
            "z": 2.3,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": -60.0,
            "width": 400,
            "height": 300,
            "fov": 100,
            "id": "rgb_left",
        },
        {
            "type": "sensor.camera.rgb",
            "x": 1.3,
            "y": 0.0,
            "z": 2.3,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 60.0,
            "width": 400,
            "height": 300,
            "fov": 100,
            "id": "rgb_right",
        },
        {
            "type": "sensor.lidar.ray_cast",
            "x": 1.3,
            "y": 0.0,
            "z": 2.5,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": -90.0,
            "id": "lidar",
        },
        {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.05,
            "id": "imu",
        },
        {
            "type": "sensor.other.gnss",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.01,
            "id": "gps",
        },
        {
            "type": "sensor.speedometer", 
            "reading_frequency": 20, 
            "id": "speed"
        },
        # {
        #     "type": 'sensor.other.collision',
        #     "id": "collision"
        # }
    ]
    ),
    policy=dict(target_speed=40, ),

)
#     env=dict(
#         simulator=dict(
#             planner=dict(type='behavior', ),
#             n_vehicles=20,
#             #n_pedestrians=25,
#             disable_two_wheels=True,
#             obs=(
#                 dict(
#                     name='rgb',
#                     type='rgb',
#                     size=[400, 400],
#                     position=[-5.5, 0, 2.8],
#                     rotation=[-15, 0, 0],
#                 ),
#                 dict(
#                     name='birdview',
#                     type='bev',
#                     size=[320, 320],
#                     pixels_per_meter=6,
#                 ),
#             ),
#             waypoint_num=50,
#             #debug=True,
#         ),
#         #no_rendering=True,
#         visualize=dict(
#             type='rgb',
#             outputs=['show']
#         ),
#     ),
#     policy=dict(target_speed=40, ),
# )

main_config = EasyDict(casezoo_config)

# def get_control(steer, throttle, brake):
#     control = carla.VehicleControl()
#     control.steer = float(steer)
#     control.throttle = float(throttle)
#     control.brake = float(brake)
#     return control

def get_vehicle_control(actions):
    control = carla.VehicleControl()
    control.steer = float(actions['steer'])
    control.throttle = float(actions['throttle'])
    control.brake = float(actions['brake'])
    return control


# def main(args, cfg, seed=0):
#     configs = []
#     if args.route is not None:
#         routes = args.route[0]
#         scenario_file = args.route[1]
#         single_route = None
#         if len(args.route) > 2:
#             single_route = args.route[2]

#         configs += RouteParser.parse_routes_file(routes, scenario_file, single_route)

#     if args.scenario is not None:
#         configs += ScenarioConfigurationParser.parse_scenario_configuration(args.scenario)

#     carla_env = CarlaEnvWrapper(ScenarioCarlaEnv(cfg.env, args.host, args.port))
#     carla_env.seed(seed)
#     set_pkg_seed(seed)
#     auto_policy = AutoPIDPolicy(cfg.policy).eval_mode

#     for config in configs:
#         auto_policy._reset([0])
#         obs = carla_env.reset(config)
#         while True:
#             actions = auto_policy._forward({0: obs})
#             action = actions[0]['action']
#             timestep = carla_env.step(action)
#             obs = timestep.obs
#             if timestep.info.get('abnormal', False):
#                 # If there is an abnormal timestep, reset all the related variables(including this env).
#                 auto_policy._reset([0])
#                 obs = carla_env.reset(config)
#             carla_env.render()
#             if timestep.done:
#                 break
#     carla_env.close()
def main(args, cfg):
    env = ScenarioCarlaEnv(cfg.env, args.host, args.port)
    index = 0
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
    while route_indexer.peek():
        config = route_indexer.next(index)
        index += 1
        obs = env.reset(config)
        auto_policy.reset([0])
        while env._running:
            if "rgb" in obs.keys() and obs['rgb'] is not None:
                show_image(obs['rgb'])
            actions = auto_policy.forward({0: obs})
            action = get_vehicle_control(actions[0]['action'])
            obs, reward, done, info = env.step(action)
        env.close() 


if __name__ == "__main__":
    description = ("DI-drive Scenario Environment")

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--route', help='Run a route as a scenario (input:(route_file,scenario_file,[route id]))', nargs='+', type=str)
    parser.add_argument('--scenario', help='Run a single scenario (input: scenario name)', type=str)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000,
                        help='TCP port to listen to (default: 9000)', type=int)
    parser.add_argument('--tm-port', default=None,
                        help='Port to use for the TrafficManager (default: None)', type=int)
    parser.add_argument('--repetitions', default=1,)

    args = parser.parse_args()
    args.routes = os.path.abspath('./core/data/srunner/routes_debug.xml')
    args.scenarios = os.path.abspath('./core/data/srunner/all_towns_traffic_scenarios1_3_4_8.json')
    args.host = "localhost"
    args.port = 2000

    main(args, main_config)
