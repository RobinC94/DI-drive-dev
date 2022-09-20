import os
import copy
import logging
import time
import numpy as np
import carla
import math
import weakref
import shapely.geometry
from enum import Enum
from easydict import EasyDict
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
from threading import Thread
from ding.utils.default_helper import deep_merge_dicts
from srunner.scenariomanager.timer import GameTime



def get_random_sample(range_list):
    res = []
    for _range in range_list:
        num = np.random.random() * _range * 2 - _range
        res.append(num)
    return res


class SensorHelper(object):
    """
    Interfaces for sensors required for vehicles and data buffer for all sensor data in Carla. The updating for Carla
    sensors are not synchronous. In each tick, the newest sensor data is obtained from sensor data buffer and returned
    to the simulator. This class provides an interface that can easily create, receive data and destroy all
    kinds of sensors in Carla according to config, and apply the same sensor augmantation to all camera sensors.

    :Arguments:
        - obs_cfg (Dict): Config dict for sensor
        - aug_cfg (Dict, optional): Config dict for sensor augmentation. Defaults to None.

    :Interfaces: setup_sensors, get_sensors_data, clear_up
    """

    def __init__(
            self,
            obs_cfg: Dict,
            aug_cfg: Optional[Dict] = None,
    ) -> None:
        self._obs_cfg = obs_cfg
        self._aug_cfg = aug_cfg
        self._sensors_dict = {}
        self._data_buffers = {}
        self._timestamps = {}
        self._random_aug_pos = None
        self._random_aug_rot = None
        self.hero_vehicle = None

    def setup_sensors(self, world: carla.World, vehicle: carla.Actor) -> None:
        """
        Create the sensors defined in config and attach them to the hero vehicle

        :Arguments:
            - world (carla.World): Carla world
            - vehicle (carla.Actor): ego vehicle
        """
        self.hero_vehicle = vehicle
        bp_library = world.get_blueprint_library()
        # if self._aug_cfg:
        #     self._aug_cfg = EasyDict(deep_merge_dicts(DEFAULT_CAMERA_AUG_CONFIG, self._aug_cfg))
        #     if min(self._aug_cfg.position_range) < 0 or min(self._aug_cfg.rotation_range) < 0:
        #         raise ValueError('Augmentation parameters must greater than 0!')
        #     self._random_aug_pos = get_random_sample(self._aug_cfg.position_range)
        #     self._random_aug_rot = get_random_sample(self._aug_cfg.rotation_range)
        # else:
        #     self._random_aug_pos = [0, 0, 0]
        #     self._random_aug_rot = [0, 0, 0]
        for obs_item in self._obs_cfg:
            # bp = bp_library.find(str(obs_item['type']))
            # print("=============obs_item['type']:",obs_item['type'])
            if obs_item.type.startswith('sensor.speedometer'):
                delta_time = world.get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(vehicle, frame_rate)
            else:
                sensor_bp = bp_library.find(str(obs_item['type']))
                if obs_item.type.startswith('sensor.camera.rgb'):
                    sensor_bp.set_attribute('image_size_x', str(obs_item['width']))
                    sensor_bp.set_attribute('image_size_y', str(obs_item['height']))
                    sensor_bp.set_attribute('fov', str(obs_item['fov']))
                    sensor_bp.set_attribute('lens_circle_multiplier', str(3.0))
                    sensor_bp.set_attribute('lens_circle_falloff', str(3.0))
                    sensor_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    sensor_bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=obs_item['x'], y=obs_item['y'],
                                                        z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                        roll=obs_item['roll'],
                                                        yaw=obs_item['yaw'])
                
                elif obs_item.type.startswith('sensor.camera.semantic_segmentation'):
                    sensor_bp.set_attribute('image_size_x', str(obs_item['width']))
                    sensor_bp.set_attribute('image_size_y', str(obs_item['height']))
                    sensor_bp.set_attribute('fov', str(obs_item['fov']))

                    sensor_location = carla.Location(x=obs_item['x'], y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                     roll=obs_item['roll'],
                                                     yaw=obs_item['yaw'])
                elif obs_item.type.startswith('sensor.camera.depth'):
                    sensor_bp.set_attribute('image_size_x', str(obs_item['width']))
                    sensor_bp.set_attribute('image_size_y', str(obs_item['height']))
                    sensor_bp.set_attribute('fov', str(obs_item['fov']))

                    sensor_location = carla.Location(x=obs_item['x'], y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                     roll=obs_item['roll'],
                                                     yaw=obs_item['yaw'])



                elif obs_item.type.startswith("sensor.lidar.ray_cast"):
                    sensor_bp.set_attribute('range', str(85))
                    sensor_bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                    sensor_bp.set_attribute('channels', str(64))
                    sensor_bp.set_attribute('upper_fov', str(10))
                    sensor_bp.set_attribute('lower_fov', str(-30))
                    sensor_bp.set_attribute('points_per_second', str(600000))
                    sensor_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    sensor_bp.set_attribute('dropoff_general_rate', str(0.45))
                    sensor_bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    sensor_bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=obs_item['x'], y=obs_item['y'],
                                                        z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                        roll=obs_item['roll'],
                                                        yaw=obs_item['yaw'])
                
                elif obs_item.type.startswith('sensor.lidar.ray_cast_semantic'):
                    sensor_bp.set_attribute('range', str(85))
                    sensor_bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 for old lidar models
                    sensor_bp.set_attribute('channels', str(64))
                    sensor_bp.set_attribute('upper_fov', str(10))
                    sensor_bp.set_attribute('lower_fov', str(-30))
                    sensor_bp.set_attribute('points_per_second', str(600000))
                    sensor_location = carla.Location(x=obs_item['x'], y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                     roll=obs_item['roll'],
                                                     yaw=obs_item['yaw'])

                elif obs_item.type.startswith('sensor.other.radar'):
                    sensor_bp.set_attribute('horizontal_fov', str(obs_item['fov']))  # degrees
                    sensor_bp.set_attribute('vertical_fov', str(obs_item['fov']))  # degrees
                    sensor_bp.set_attribute('points_per_second', '1500')
                    sensor_bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=obs_item['x'],
                                                     y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                     roll=obs_item['roll'],
                                                     yaw=obs_item['yaw'])

                elif obs_item['type'].startswith('sensor.other.gnss'):
                    # bp.set_attribute('noise_alt_stddev', str(0.000005))
                    # bp.set_attribute('noise_lat_stddev', str(0.000005))
                    # bp.set_attribute('noise_lon_stddev', str(0.000005))
                    sensor_bp.set_attribute('noise_alt_bias', str(0.0))
                    sensor_bp.set_attribute('noise_lat_bias', str(0.0))
                    sensor_bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=obs_item['x'],
                                                     y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation()

                elif obs_item['type'].startswith('sensor.other.imu'):
                    sensor_bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    sensor_bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    sensor_bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    sensor_bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    sensor_bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    sensor_bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=obs_item['x'],
                                                     y=obs_item['y'],
                                                     z=obs_item['z'])
                    sensor_rotation = carla.Rotation(pitch=obs_item['pitch'],
                                                     roll=obs_item['roll'],
                                                     yaw=obs_item['yaw'])


                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
            sensor.listen(CallBack(obs_item.id, obs_item.type, self))
            self.register_sensor(obs_item.id, sensor)

    def clean_up(self) -> None:
        """
        Remove and destroy all sensors
        """
        for key in self._sensors_dict:
            if self._sensors_dict[key] is not None:
                if self._sensors_dict[key].is_alive:
                    self._sensors_dict[key].stop()
                    self._sensors_dict[key].destroy()
                self._sensors_dict[key] = None
        time.sleep(0.1)
        self._sensors_dict.clear()
        self._data_buffers.clear()
        self._timestamps.clear()

    def register_sensor(self, tag: str, sensor: Any) -> None:
        """
        Registers the sensors
        """
        if tag in self._sensors_dict:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_dict[tag] = sensor
        self._data_buffers[tag] = None
        self._timestamps[tag] = -1

    def update_sensor(self, tag: str, data: Any, timestamp: Any) -> None:
        """
        Updates the sensor
        """
        if tag not in self._sensors_dict:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))
        # print("tag:",tag)
        self._data_buffers[tag] = data
        self._timestamps[tag] = timestamp

    def all_sensors_ready(self) -> bool:
        """
        Checks if all the sensors have sent data at least once
        """
        for key in self._sensors_dict:
            if self._data_buffers[key] is None:
                return False
        return True

    def get_sensors_data(self) -> Dict:
        """
        Get all registered sensor data from buffer

        :Returns:
            Dict: all newest sensor data
        """
        sensor_data = {}
        for obs_item in self._obs_cfg:
            if obs_item.type == 'sensor.camera.rgb':
                key = obs_item.id
                img = self._data_buffers[key]
                sensor_data[key] = img
            elif obs_item.type == 'sensor.lidar.ray_cast':
                key = obs_item.id
                lidar = self._data_buffers[key]
                sensor_data[key] = lidar
            elif obs_item.type == 'sensor.speedometer':
                key = obs_item.id
                speedometer = self._data_buffers[key]
                sensor_data[key] = speedometer
            elif obs_item.type == 'sensor.camera.semantic_segmentation':
                key = obs_item.id
                camera = self._data_buffers[key]
                sensor_data[key] = camera
            elif obs_item.type == 'sensor.camera.depth':
                key = obs_item.id
                camera = self._data_buffers[key]
                sensor_data[key] = camera
            elif obs_item.type == 'sensor.lidar.ray_cast_semantic':
                key = obs_item.id
                lidar = self._data_buffers[key]
                sensor_data[key] = lidar  
            elif obs_item.type == 'sensor.other.radar':
                key = obs_item.id
                radar = self._data_buffers[key]
                sensor_data[key] = radar                         
            elif obs_item.type == 'sensor.other.gnss':
                key = obs_item.id
                gnss = self._data_buffers[key]
                sensor_data[key] = gnss   
            elif obs_item.type == 'sensor.other.imu':
                key = obs_item.id
                imu = self._data_buffers[key]
                sensor_data[key] = imu     
            # elif obs_item.type in ['rgb', 'segmentation', 'lidar', 'gnss']:
            #     key = obs_item.name
            #     img = self._data_buffers[key]
            #     sensor_data[key] = img
            # elif obs_item.type == 'depth':
            #     key = obs_item.name
            #     raw = self._data_buffers[key]
            #     img = raw.astype(np.float64)
            #     R = img[..., 0]
            #     G = img[..., 1]
            #     B = img[..., 2]
            #     depth = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            #     depth = 1000 * depth
            #     sensor_data[key] = depth
        # if self._aug_cfg:
        #     sensor_data['aug'] = {
        #         'aug_pos': np.array(self._random_aug_pos),
        #         'aug_rot': np.array(self._random_aug_rot),
        #     }
        return sensor_data

# 自己写的：解析img, lidar, gnss
class CallBack(object):
    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag: str, type: str, wrapper: Any) -> None:
        """
        Initializes the call back
        """
        self._tag = tag
        self._type = type
        self._data_provider = wrapper

    def __call__(self, data: Any) -> None:
        """
        call function
        """
        # print("data:",data)
        # print("type(data):",type(data))
        # if isinstance(data, carla.Image):
        #     self._parse_image_cb(data, self._tag)
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
            self._parse_semantic_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')
    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_semantic_lidar_cb(self, semantic_lidar_data, tag):
        points = np.frombuffer(semantic_lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        self._data_provider.update_sensor(tag, points, semantic_lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper

class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}



