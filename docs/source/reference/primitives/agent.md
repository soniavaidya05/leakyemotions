# Agent

```{eval-rst}
.. autoclass:: agentarium.primitives.Agent
```

## Methods

### Abstract Methods
```{eval-rst}
.. automethod:: agentarium.primitives.Agent.reset
.. automethod:: agentarium.primitives.Agent.pov
.. automethod:: agentarium.primitives.Agent.get_action
.. automethod:: agentarium.primitives.Agent.act
.. automethod:: agentarium.primitives.Agent.is_done
```
### Non-Abstract Methods
```{eval-rst}
.. automethod:: agentarium.primitives.Agent.add_memory
.. automethod:: agentarium.primitives.Agent.transition
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
