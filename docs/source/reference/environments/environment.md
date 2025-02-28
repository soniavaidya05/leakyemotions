# GridWorldEnv

```{eval-rst}
.. autoclass:: agentarium.environments.GridworldEnv
```

## Methods
### Class Methods
```{eval-rst}
.. automethod:: agentarium.environments.GridworldEnv.create_world
.. automethod:: agentarium.environments.GridworldEnv.add
.. automethod:: agentarium.environments.GridworldEnv.remove
.. automethod:: agentarium.environments.GridworldEnv.move
.. automethod:: agentarium.environments.GridworldEnv.observe
.. automethod:: agentarium.environments.GridworldEnv.take_turn
.. automethod:: agentarium.environments.GridworldEnv.valid_location
```

### Static Methods
```{eval-rst}
.. tip::
    You can still call a static method using an instance.

.. automethod:: agentarium.environments.GridworldEnv.get_entities_of_kind
```

## Attributes
```{eval-rst}
.. autoattribute:: agentarium.environments.GridworldEnv.height

    The height of the gridworld.

.. autoattribute:: agentarium.environments.GridworldEnv.width

    The width of the gridworld.

.. autoattribute:: agentarium.environments.GridworldEnv.layers

    The number of layers that the gridworld has.

.. autoattribute:: agentarium.environments.GridworldEnv.default_entity

    An entity that the gridworld is filled with at creation by default.
    
.. autoattribute:: agentarium.environments.GridworldEnv.world

    A representation of the gridworld as a Numpy array of Entities, with dimensions height x width x layers.
    
.. autoattribute:: agentarium.environments.GridworldEnv.turn

    The number of turns taken by the environment.
```
