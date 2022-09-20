from typing import Optional
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import carla


class CarlaDataProviderExpand(object):
    # CarlaDataProvider的拓展静态类，用于获取信息
    _carla_actor_pool = dict()
    _actor_speed_vector_map = dict()
    _actor_acceleration_map = dict()
    _actor_angular_velocity_map = dict()

    @staticmethod
    def prepare_info_map():
        for id, actor in CarlaDataProvider.get_actors():
            # print("actor:", actor)
            if actor in CarlaDataProviderExpand._actor_speed_vector_map:
                # raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
                pass
            else:
                CarlaDataProviderExpand._actor_speed_vector_map[actor] = None
    
            if actor in CarlaDataProviderExpand._actor_acceleration_map:
                # raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
                pass
            else:
                CarlaDataProviderExpand._actor_acceleration_map[actor] = None
            
            if actor in CarlaDataProviderExpand._actor_angular_velocity_map:
                # raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
                pass
            else:
                CarlaDataProviderExpand._actor_angular_velocity_map[actor] = None


    @staticmethod
    def get_speed_vector(actor: carla.Actor) -> Optional[carla.Vector3D]:
        """
        returns the absolute speed for the given actor
        """
        for key in CarlaDataProviderExpand._actor_speed_vector_map:
            if key.id == actor.id:
                return CarlaDataProviderExpand._actor_speed_vector_map[key]
        print('WARNING: {}.get_speed: {} not found!'.format(__name__, actor))
        return None

    @staticmethod
    def get_acceleration(actor: carla.Actor) -> Optional[carla.Vector3D]:
        """
        returns the absolute acceleration for the given actor
        """
        for key in CarlaDataProviderExpand._actor_acceleration_map:
            if key.id == actor.id:
                return CarlaDataProviderExpand._actor_acceleration_map[key]
        print('WARNING: {}.get_acceleration: {} not found!'.format(__name__, actor))
        return None

    @staticmethod
    def get_angular_velocity(actor: carla.Actor) -> Optional[carla.Vector3D]:
        """
        returns the angular velocity for the given actor
        """
        for key in CarlaDataProviderExpand._actor_angular_velocity_map:
            if key.id == actor.id:
                return CarlaDataProviderExpand._actor_angular_velocity_map[key]
        print('WARNING: {}.get_angular_velocity: {} not found!'.format(__name__, actor))
        return None



    @staticmethod
    def on_carla_tick() -> None:
        """
        Callback from CARLA
        """
        CarlaDataProviderExpand.prepare_info_map()
        for actor in CarlaDataProviderExpand._actor_speed_vector_map:
            if actor is not None and actor.is_alive:
                CarlaDataProviderExpand._actor_speed_vector_map[actor] = actor.get_velocity()
                # print("actor:",actor.id," velocity:", actor.get_velocity())



        for actor in CarlaDataProviderExpand._actor_acceleration_map:
            if actor is not None and actor.is_alive:
                CarlaDataProviderExpand._actor_acceleration_map[actor] = actor.get_acceleration()

        for actor in CarlaDataProviderExpand._actor_angular_velocity_map:
            if actor is not None and actor.is_alive:
                CarlaDataProviderExpand._actor_angular_velocity_map[actor] = actor.get_angular_velocity()



        world = CarlaDataProvider._world
        if world is None:
            print("WARNING: CarlaInterface couldn't find the world")
