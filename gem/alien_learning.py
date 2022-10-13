from gem.environment.elements.alien_elements import Agent
from models.simple_dqn import Model_simple_linear_DQN
import torch


save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def create_models():

    models = []
    models.append(
        Model_simple_linear_DQN(
            lr=0.0001,
            replay_size=1024,  
            in_size=4,  
            hid_size1=10,  
            hid_size2=10,  
            out_size=2,
            priority_replay=False,
            device=device,
        )
    )  # agent model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models


models = create_models()

agent = Agent(0)

done = 0
approaches = [0,0]
losses = 0

alien_type, appearence, cooperation = agent.generate_alien()
appearence = torch.tensor(appearence).float().to(device)
for epoch in range(20000):

    state = appearence

    action = models[0].take_action([appearence, .1])
    reward = agent.transition(action, cooperation)

    approaches[alien_type] = approaches[alien_type] + action

    alien_type, appearence, cooperation = agent.generate_alien()
    appearence = torch.tensor(appearence).float().to(device)

    exp = [1, (
        state,
        action,
        reward,
        appearence,
        done,
    )]

    agent.episode_memory.append(exp)
    loss = models[0].training(exp)
    losses = losses + loss.detach().cpu().numpy()


    if epoch % 100 == 0:
        print("epoch:" , epoch, "loss: ",losses/5000, "approaches: ", approaches)
        approaches = [0,0]
        losses = 0




