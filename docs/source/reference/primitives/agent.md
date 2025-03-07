# Agent

```{eval-rst}
.. autoclass:: sorrel.agents.Agent
```

## Methods

### Abstract Methods
```{eval-rst}
.. automethod:: sorrel.agents.Agent.reset
.. automethod:: sorrel.agents.Agent.pov
.. automethod:: sorrel.agents.Agent.get_action
.. automethod:: sorrel.agents.Agent.act
.. automethod:: sorrel.agents.Agent.is_done
```
### Non-Abstract Methods
```{eval-rst}
.. automethod:: sorrel.agents.Agent.add_memory
.. automethod:: sorrel.agents.Agent.transition
```

## Attributes
```{eval-rst}
        
.. autoattribute:: sorrel.agents.Agent.observation_spec

    The observation specification to use for this agent.
    
.. autoattribute:: sorrel.agents.Agent.model

    The model that this agent uses.
    
.. autoattribute:: sorrel.agents.Agent.action_space

    The set of actions that the agent is able to take.
```
