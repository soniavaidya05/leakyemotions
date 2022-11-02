from gem.environment.elements import EmptyObject, TagAgent, Wall
from models.perception import agent_visualfield
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
#                      Tag Agent transition rules
# ---------------------------------------------------------------------


def tagAgentTransitions(
    holdObject,
    action,
    world,
    models,
    i,
    j,
    totalRewards,
    done,
    input,
    update_experience_buffer=True,
):

    new_locaton_1 = i
    new_locaton_2 = j

    reward = 0

    if action == 0:
        attempted_locaton_1 = i - 1
        attempted_locaton_2 = j
    elif action == 1:
        attempted_locaton_1 = i + 1
        attempted_locaton_2 = j
    elif action == 2:
        attempted_locaton_1 = i
        attempted_locaton_2 = j - 1
    elif action == 3:
        attempted_locaton_1 = i
        attempted_locaton_2 = j + 1

    if holdObject.frozen > 0:
        # if object is frozen, just decrease frozen count and then go from there
        # hope this doesn't break learning
        holdObject.dethaw()
    elif world[attempted_locaton_1, attempted_locaton_2, 0].passable == 1:
        world[i, j, 0] = EmptyObject()
        reward += 1
        world[attempted_locaton_1, attempted_locaton_2, 0] = holdObject
        new_locaton_1 = attempted_locaton_1
        new_locaton_2 = attempted_locaton_2
        totalRewards += 1
    elif world[attempted_locaton_1, attempted_locaton_2, 0].kind == "TagAgent":
        # bump into each other
        alterAgent = world[attempted_locaton_1, attempted_locaton_2, 0]
        if holdObject.is_it == 1 and alterAgent.is_it == 0:
            reward += 7
            holdObject.tag()  # change who is it
            alterAgent.tag()

            lastexp = alterAgent.replay[-1]
            exp = (lastexp[0], lastexp[1], -15, lastexp[3], 1)
            world[attempted_locaton_1, attempted_locaton_2, 0].replay.append(exp)

        elif holdObject.is_it == 0 and alterAgent.is_it == 1:
            # agent bumpbed into tag person

            reward -= 13
            holdObject.tag()  # change who is it
            alterAgent.tag()

        elif holdObject.is_it == alterAgent.is_it:
            # bump into each other
            reward -= 0.1

    elif world[attempted_locaton_1, attempted_locaton_2, 0].kind == "wall":
        reward = -0.1

    if update_experience_buffer == True:
        img2 = agent_visualfield(
            world, (new_locaton_1, new_locaton_2), holdObject.appearance.shape, holdObject.vision
        )
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[new_locaton_1, new_locaton_2, 0].replay.append(exp)
        world[new_locaton_1, new_locaton_2, 0].reward += reward

    return world, models, totalRewards
