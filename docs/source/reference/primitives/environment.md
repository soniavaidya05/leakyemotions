# GridWorldEnv

```{eval-rst}
.. autoclass:: agentarium.primitives.GridworldEnv
```

## Methods
### Class Methods
```{eval-rst}
.. automethod:: agentarium.primitives.GridworldEnv.add
.. automethod:: agentarium.primitives.GridworldEnv.remove
.. automethod:: agentarium.primitives.GridworldEnv.move
.. automethod:: agentarium.primitives.GridworldEnv.observe
.. automethod:: agentarium.primitives.GridworldEnv.valid_location
```

### Static Methods
```{eval-rst}
.. tip::
    You can still call a static method using an instance.

.. automethod:: agentarium.primitives.GridworldEnv.get_entities_of_kind
```

## Attributes
```{eval-rst}
.. autoattribute:: agentarium.primitives.GridworldEnv.height

    The height of the gridworld.

.. autoattribute:: agentarium.primitives.GridworldEnv.width

    The width of the gridworld.

.. autoattribute:: agentarium.primitives.GridworldEnv.layers

    The number of layers that the gridworld has.

.. autoattribute:: agentarium.primitives.GridworldEnv.default_entity

    The entity that the gridworld is filled with at creation by default.
    
.. autoattribute:: agentarium.primitives.GridworldEnv.world

    A representation of the gridworld as a Numpy array of Entities, with dimensions height x width x layers.
```
