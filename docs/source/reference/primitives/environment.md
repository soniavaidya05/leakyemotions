# GridWorldEnv

```{eval-rst}
.. autoclass:: agentarium.primitives.GridworldEnv
```

## Methods
```{eval-rst}
.. automethod:: agentarium.primitives.GridworldEnv.add
.. automethod:: agentarium.primitives.GridworldEnv.remove
.. automethod:: agentarium.primitives.GridworldEnv.move
.. automethod:: agentarium.primitives.GridworldEnv.observe
.. automethod:: agentarium.primitives.GridworldEnv.valid_location
```

TODO: formalize get_entities() and get_entities_()

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

    A representation of the gridworld as an Numpy array of Entities, with dimensions height x width x layers.
```
