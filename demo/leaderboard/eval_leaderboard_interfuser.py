from easydict import EasyDict
import carla

import argparse

from core.envs.scenario_leaderboard_env import ScenarioLeaderboardEnv
from core.policy.interfuser_policy import InterfuserPolicy
from leaderboard.utils.statistics_manager import StatisticsManager


cfg = dict(
    arguments=
        dict(
            # 需要写成绝对路径，否则配置环境变量会失败
            checkpoint='/home/wuche/sense-lab/xad_0819/xad/results/results.json', \
            debug=0, 
            host='localhost', 
            port='2000', 
            record='', 
            repetitions=1, 
            resume=True, 
            routes='/home/wuche/sense-lab/xad_0819/xad/core/data/srunner/routes_town05_long.xml', \
            scenarios='/home/wuche/sense-lab/xad_0819/xad/core/data/srunner/all_towns_traffic_scenarios1_3_4.json', \
            timeout='6000.0', 
            track='SENSORS', 
            trafficManagerPort='2500', 
            trafficManagerSeed='0',
            savepath='.'
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
    ]
)
cfg = EasyDict(cfg)
controller_dict = {
    # Controller
    "turn_KP" : 1.25,
    "turn_KI" : 0.75,
    "turn_KD" : 0.3,
    "turn_n" : 40,  # buffer size
    "speed_KP" : 5.0,
    "speed_KI" : 0.5,
    "speed_KD" : 1.0,
    "speed_n" : 40,  # buffer size
    "max_throttle" : 0.75,  # upper limit on throttle signal value in dataset
    "brake_speed" : 0.1,  # desired speed below which brake is triggered
    "brake_ratio" : 1.1 , # ratio of speed to desired speed at which brake is triggered
    "clip_delta" : 0.35,  # maximum change in speed input to logitudinal controller
    "max_speed" : 5,
    "collision_buffer" : [2.5, 1.2],
    "model_path" : "/home/wuche/sense-lab/xad_0819/xad/core/data/interfuser.pth.tar",
    "momentum" : 0,
    "skip_frames" : 1,
    "detect_threshold" : 0.04,
    "model" : "interfuser_baseline_seperate_all"
}

def main(args):
    cfg.host = args.host
    cfg.port = args.port
    cfg.trafficManagerPort = args.tmport
    env=ScenarioLeaderboardEnv(cfg=cfg)
    index = 0
    while env.route_indexer.peek():
        config = env.route_indexer.next(index)
        index += 1
        obs=env.reset(cfg, config)
        policy_cfg={"world": env.world, "config": config}
        policy_cfg.update(controller_dict)
        policy_cfg=EasyDict(policy_cfg)
        policy=InterfuserPolicy(policy_cfg)

        while env._running:
            timestamp = env.tick_carla_world()
            if timestamp and env._running:
                action=policy.run_step(obs, timestamp)
                obs, reward, done, info = env.step(action)
            else:
                break
            env.adjust_world_transform()

        env.route_indexer.save_state(cfg.arguments.checkpoint)
        env.close() 

    # save global statistics
    print("\033[1m> Registering the global statistics\033[0m")
    global_stats_record = env.statistics_manager.compute_global_statistics(env.route_indexer.total)
    StatisticsManager.save_global_record(global_stats_record, env.sensor_icons, env.route_indexer.total, cfg.arguments.checkpoint)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='leaderboard eval')
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000,
                        help='TCP port to listen to (default: 9000)', type=int)
    parser.add_argument('--tmport', default=2500,
                        help='Port to use for the TrafficManager (default: None)', type=int)

    args = parser.parse_args()
    main(args)
