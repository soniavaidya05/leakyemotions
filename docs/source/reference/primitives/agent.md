# Agent

```{eval-rst}
.. autoclass:: agentarium.primitives.Agent
```

## Methods
Note that the following methods must be implemented in any implementation of this abstract class.

```{eval-rst}
.. automethod:: agentarium.primitives.Agent.act
.. automethod:: agentarium.primitives.Agent.pov
.. automethod:: agentarium.primitives.Agent.transition
.. automethod:: agentarium.primitives.Agent.reset
```
TODO: get an answer on add_memory()

## Attributes
```{eval-rst}
        
.. autoattribute:: agentarium.primitives.Agent.cfg

    The configuration to use for this agent.
    
.. autoattribute:: agentarium.primitives.Agent.model

    The model that this agent uses.
    
.. autoattribute:: agentarium.primitives.Agent.action_space

    The set of actions that the agent is able to take.
```
