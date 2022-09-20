import metadrive
from easydict import EasyDict
from functools import partial

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from core.envs import DriveEnvWrapper
from demo.metadrive.model import ConvQAC

metadrive_basic_config = dict(
    exp_name='metadrive_basic_sac_eval',
    env=dict(
        metadrive=dict(use_render=True),
        manager=dict(),
        n_evaluator_episode=1,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 84, 84],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=1000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ),
    )
)

main_config = EasyDict(metadrive_basic_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(TopDownMetaDrive(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        SACPolicy,
    )

    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvQAC(**cfg.policy.model)
    policy = SACPolicy(cfg.policy, model=model)

    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, None, exp_name=cfg.exp_name
    )

    stop, rate = evaluator.eval()

    evaluator.close()


if __name__ == '__main__':
    main(main_config)
