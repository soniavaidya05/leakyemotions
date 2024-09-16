import torch
from typing import Optional, Sequence

from examples.trucks.agents import Memory
from gem.models.ann import ANN
from gem.primitives import Object, GridworldEnv
from gem.utils import visual_field

class Agent(Object):
  """Cleanup agent."""

  def __init__(self, cfg, appearance, model):

    super().__init__(appearance)

    self.vision = cfg.agent.agent.vision
    self.direction = 0 # 90 degree rotation: default at 0 degrees
    self.action_space = [0, 1, 2, 3, 4, 5, 6]
    has_transitions = True

    # training-related features
    self.action_type = "neural_network"
    self.model = model
    self.episode_memory = Memory(cfg.agent.agent.memory_size)
    self.num_memories = cfg.agent.agent.memory_size
    self.init_rnn_state = None

    # logging features
    self.outcome_record = {
      "harvest": 0,
      "zap": 0,
      "get_zapped": 0,
      "clean": 0
    }

  def init_replay(self) -> None:

    """Fill in blank images for the LSTM."""

    priority = torch.tensor(0.1)
    num_frames = self.model.num_frames
    if self.cfg.full_mdp:
      state = torch.zeros(1, num_frames, *self.model.state_size).float()
    else:
      # Number of one-hot code channels
      C = len(self.appearance)
      H = W = self.vision * 2 + 1
      state = torch.zeros(1, num_frames, C, H, W).float()

    action = torch.tensor(7.0) # Action outside the action space
    reward = torch.tensor(0.0)
    done = torch.tensor()
    exp = (priority, (state, action, reward, state, done))
    self.episode_memory.append(exp)

  def movement(self, action: int) -> tuple[int, ...]:
    
    # Default location
    next_location = self.location

    # Define the forward (index 0) and back (index 1) movements
    match(self.direction):
      case 0: # UP
        shifts = [(-1, 0), (1, 0)]
      case 1: # RIGHT
        shifts = [(0, 1), (0, -1)]
      case 2: # DOWN
        shifts = [(1, 0), (-1, 0)]
      case 3: # LEFT
        shifts = [(0, -1), (0, 1)]

    if action == 0: # NOOP
      pass
      
    if action == 1: # FORWARD
      shift = shifts[0]
      next_location = (self.location[0] + shift[0], self.location[1] + shift[1], self.location[2])
    
    if action == 2: # BACK
      shift = shifts[2]
      next_location = (self.location[0] + shift[0], self.location[1] + shift[1], self.location[2])

    if action == 3: # TURN CLOCKWISE
      # Add 90 degrees; modulo 4 to ensure range of [0, 1, 2, 3]   
      self.direction = (self.direction + 1) % 4
    
    if action == 4: # TURN COUNTERCLOCKWISE
      self.direction = (self.direction - 1) % 4

    return next_location
  
  def pov(self,
        env) -> torch.Tensor:
    '''
    Defines the agent's observation function
    '''
    # Get the previous state
    previous_state = self.episode_memory.get_last_memory('states')

    # Get the frames from the previous state
    current_state = previous_state.clone()

    current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

    # If the environment is a full MDP, get the whole world image
    if env.full_mdp:
        image = visual_field(env.world, env.color_map, channels=self.cfg.model.iqn.parameters.state_size[0])
    # Otherwise, use the agent observation function
    else:
        image = visual_field(env.world, env.color_map, self.location, self.vision, channels=self.cfg.model.iqn.parameters.state_size[0])

    # Update the latest state to the observation
    state_now = torch.tensor(image).unsqueeze(0)
    current_state[:, -1, :, :, :] = state_now

    return current_state
  
  def transition(self, env: GridworldEnv):
    """Changes the world based on action taken."""
            # Get current state
    state = self.pov(env)
    reward = 0

    # Take action based on current state
    action = self.model.take_action(state)

    # Attempt the transition 
    attempted_location = self.movement(action)
    target_object = env.observe(attempted_location)
    env.move(self, attempted_location)

    # Get the interaction reward
    reward += target_object.value

    # Add to the encounter record
    if str(target_object) in self.encounters.keys():
        self.encounters[str(target_object)] += 1 

    # Get the next state   
    next_state = self.pov(env)

    return state, action, reward, next_state, False


