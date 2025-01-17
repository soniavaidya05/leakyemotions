# Entity

```{eval-rst}
.. autoclass:: agentarium.primitives.Entity
```

## Methods
```{eval-rst}
.. automethod:: agentarium.primitives.Entity.transition
```

## Attributes
```{eval-rst}
.. autoattribute:: agentarium.primitives.Entity.location

    The location of the object. It may take on the value of None when the Entity is first initialized.
    
.. autoattribute:: agentarium.primitives.Entity.value

    The reward provided to an agent upon interaction. It is 0 by default.
    
.. autoattribute:: agentarium.primitives.Entity.passable

    Whether the object can be traversed by an agent. It is False by default.
    
.. autoattribute:: agentarium.primitives.Entity.has_transitions

    Whether the object has unique physics interacting with the environment. It is False by default.
    
.. autoattribute:: agentarium.primitives.Entity.kind

    The class string of the object.

```
