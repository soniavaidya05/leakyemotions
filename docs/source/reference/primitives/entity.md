# Entity

```{eval-rst}
.. autoclass:: agentarium.entities.Entity
```

## Methods
```{eval-rst}
.. automethod:: agentarium.entities.Entity.transition
```

## Attributes
```{eval-rst}
.. autoattribute:: agentarium.entities.Entity.location

    The location of the object. It may take on the value of None when the Entity is first initialized.
    
.. autoattribute:: agentarium.entities.Entity.value

    The reward provided to an agent upon interaction. It is 0 by default.
    
.. autoattribute:: agentarium.entities.Entity.passable

    Whether the object can be traversed by an agent. It is False by default.
    
.. autoattribute:: agentarium.entities.Entity.has_transitions

    Whether the object has unique physics interacting with the environment. It is False by default.
    
.. autoattribute:: agentarium.entities.Entity.kind

    The class string of the object.

```
