core.policy
#############

.. currentmodule:: core.policy

.. automodule:: core.policy


BaseCarlaPolicy
================
.. autoclass:: core.policy.base_carla_policy.BaseCarlaPolicy
    :members:


AutoPIDPolicy
================
.. autoclass:: AutoPIDPolicy
    :members: _forward_collect, _forward_eval, _reset_collect, _reset_eval


AutoMPCPolicy
================
.. autoclass:: AutoMPCPolicy
    :members: _forward_collect, _forward_eval, _reset_collect, _reset_eval


CILRSPolicy
=============
.. autoclass:: CILRSPolicy
    :members:


LBCBirdviewPolicy
====================
.. autoclass:: LBCBirdviewPolicy
    :members: _forward_eval, _reset_eval


LBCImagePolicy
====================
.. autoclass:: LBCImagePolicy
    :members: _forward_eval, _reset_eval