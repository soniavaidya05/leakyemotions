# Agent

```{eval-rst}
.. autoclass:: agentarium.agents.Agent
```

## Methods

### Abstract Methods
```{eval-rst}
.. automethod:: agentarium.agents.Agent.reset
.. automethod:: agentarium.agents.Agent.pov
.. automethod:: agentarium.agents.Agent.get_action
.. automethod:: agentarium.agents.Agent.act
.. automethod:: agentarium.agents.Agent.is_done
```
### Non-Abstract Methods
```{eval-rst}
.. automethod:: agentarium.agents.Agent.add_memory
.. automethod:: agentarium.agents.Agent.transition
```

## Attributes
```{eval-rst}
        
.. autoattribute:: agentarium.agents.Agent.observation_spec

    The observation specification to use for this agent.
    
.. autoattribute:: agentarium.agents.Agent.model

    The model that this agent uses.
    
.. autoattribute:: agentarium.agents.Agent.action_space

    The set of actions that the agent is able to take.
```
