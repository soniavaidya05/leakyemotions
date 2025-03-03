# Base model

```{eval-rst}
.. autoclass:: agentarium.models.base_model.AgentariumModel
```

## Methods

### Abstract Methods
```{eval-rst}
.. automethod:: agentarium.models.base_model.AgentariumModel.take_action
```
### Non-Abstract Methods
```{eval-rst}
.. automethod:: agentarium.models.base_model.AgentariumModel.train_step
.. automethod:: agentarium.models.base_model.AgentariumModel.set_epsilon
.. automethod:: agentarium.models.base_model.AgentariumModel.start_epoch_action
.. automethod:: agentarium.models.base_model.AgentariumModel.end_epoch_action
```

## Properties
```{eval-rst}
.. automethod:: agentarium.models.base_model.AgentariumModel.model_name
```

## Attributes
```{eval-rst}
        
.. autoattribute:: agentarium.models.base_model.AgentariumModel.input_size

    The size of the input for the model.
    
.. autoattribute:: agentarium.models.base_model.AgentariumModel.action_space

    The action space for the model.
    
.. autoattribute:: agentarium.models.base_model.AgentariumModel.memory

    The replay buffer for the model.

.. autoattribute:: agentarium.models.base_model.AgentariumModel.epsilon

    The epsilon value for the model.
```
