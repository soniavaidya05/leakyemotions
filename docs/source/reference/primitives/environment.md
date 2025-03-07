# GridWorldEnv

```{eval-rst}
.. autoclass:: sorrel.environments.GridworldEnv
```

## Methods
### Class Methods
```{eval-rst}
.. automethod:: sorrel.environments.GridworldEnv.create_world
.. automethod:: sorrel.environments.GridworldEnv.add
.. automethod:: sorrel.environments.GridworldEnv.remove
.. automethod:: sorrel.environments.GridworldEnv.move
.. automethod:: sorrel.environments.GridworldEnv.observe
.. automethod:: sorrel.environments.GridworldEnv.take_turn
.. automethod:: sorrel.environments.GridworldEnv.valid_location
```

### Static Methods
```{eval-rst}
.. tip::
    You can still call a static method using an instance.

.. automethod:: sorrel.environments.GridworldEnv.get_entities_of_kind
```

## Attributes
```{eval-rst}
.. autoattribute:: sorrel.environments.GridworldEnv.height

    The height of the gridworld.

.. autoattribute:: sorrel.environments.GridworldEnv.width

    The width of the gridworld.

.. autoattribute:: sorrel.environments.GridworldEnv.layers

    The number of layers that the gridworld has.

.. autoattribute:: sorrel.environments.GridworldEnv.default_entity

    An entity that the gridworld is filled with at creation by default.
    
.. autoattribute:: sorrel.environments.GridworldEnv.world

    A representation of the gridworld as a Numpy array of Entities, with dimensions height x width x layers.
    
.. autoattribute:: sorrel.environments.GridworldEnv.turn

    The number of turns taken by the environment.
```
