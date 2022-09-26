import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

import sys

import carla
from core.envs import ScenarioCarlaEnv, CarlaEnvWrapper
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed, deep_merge_dicts
from core.utils.others.image_helper import show_image

from leaderboard.utils.route_indexer import RouteIndexer


srunner_config = dict(
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

main_config = EasyDict(srunner_config)


def main(args, cfg):
    env = CarlaEnvWrapper(ScenarioCarlaEnv(cfg.env, args.host, args.port))
    index = 0
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
    while route_indexer.peek():
        config = route_indexer.next()
        index += 1
        obs = env.reset(config)
        auto_policy.reset([0])
        while env.running:
            if "rgb" in obs.keys() and obs['rgb'] is not None:
                show_image(obs['rgb']/255.)
            actions = auto_policy.forward({0: obs})
            obs, reward, done, info = env.step(actions[0]['action'])
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
    args.scenarios = os.path.abspath('./core/data/srunner/all_towns_traffic_scenarios.json')
    args.host = "localhost"
    args.port = 2000

    main(args, main_config)
