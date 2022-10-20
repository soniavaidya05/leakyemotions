from gem.environment.elements.alien_elements import Agent
from models.fast_slow_dqn import Model_simple_linear_DQN
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
            in_size=11,  
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
approaches = [0,0,0,0,0,0]
losses = 0


for epoch in range(1000000):
    alien_type, appearance, cooperation = agent.generate_alien()
    appearance = torch.tensor(appearance).float().to(device)

    action = models[0].take_action([appearance, .1])
    reward = agent.transition(action, cooperation, "partial")

    approaches[alien_type] = approaches[alien_type] + action

    exp = [1, (
        appearance,
        action,
        reward,
        appearance,
        done,
    )]

    agent.episode_memory.append(exp)
    loss = models[0].training(exp)
    losses = losses + loss.detach().cpu().numpy()


    if epoch % 500 == 0:
        print("epoch:" , epoch, "loss: ",losses/100, "approaches (good, bad): ", approaches)
        approaches = [0,0,0,0,0,0]
        losses = 0




