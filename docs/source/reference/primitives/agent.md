# Agent

```{eval-rst}
.. autoclass:: agentarium.primitives.Agent
```

## Methods

```{eval-rst}
.. note::
    The following methods are provided for you. You might override them in your own implementation if necessary.

.. automethod:: agentarium.primitives.Agent.add_memory

.. note::
    The following methods are abstract and therefore must be implemented in any class that inherits this class.

.. automethod:: agentarium.primitives.Agent.act
.. automethod:: agentarium.primitives.Agent.pov
.. automethod:: agentarium.primitives.Agent.transition
.. automethod:: agentarium.primitives.Agent.reset
```

## Attributes
```{eval-rst}
        
.. autoattribute:: agentarium.primitives.Agent.cfg

    The configuration to use for this agent.
    
.. autoattribute:: agentarium.primitives.Agent.model

    The model that this agent uses.
    
.. autoattribute:: agentarium.primitives.Agent.action_space

    The set of actions that the agent is able to take.
```
