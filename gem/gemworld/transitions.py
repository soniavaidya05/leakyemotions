from gem.environment.elements import EmptyObject, deadAgent
from models.perception import agentVisualField
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
#                      Agent transition rules
# ---------------------------------------------------------------------


def agentTransitions(
    holdObject, action, world, models, i, j, gamePoints, done, input, expBuff=True
):

    newLoc1 = i
    newLoc2 = j

    reward = 0

    if action == 0:
        attLoc1 = i - 1
        attLoc2 = j

    if action == 1:
        attLoc1 = i + 1
        attLoc2 = j

    if action == 2:
        attLoc1 = i
        attLoc2 = j - 1

    if action == 3:
        attLoc1 = i
        attLoc2 = j + 1

    if world[attLoc1, attLoc2, 0].passable == 1:
        world[i, j, 0] = EmptyObject()
        reward = world[attLoc1, attLoc2, 0].value
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        gamePoints[0] = gamePoints[0] + reward
    else:
        if world[attLoc1, attLoc2, 0].kind == "wall":
            reward = -0.1

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[newLoc1, newLoc2, 0].replay.append(exp)
        world[newLoc1, newLoc2, 0].reward += reward

    return world, models, gamePoints


# ---------------------------------------------------------------------
#                      Wolf transition rules
# ---------------------------------------------------------------------


def wolfTransitions(
    holdObject, action, world, models, i, j, gamePoints, done, input, expBuff=True
):

    newLoc1 = i
    newLoc2 = j

    reward = 0

    if action == 0:
        attLoc1 = i - 1
        attLoc2 = j

    if action == 1:
        attLoc1 = i + 1
        attLoc2 = j

    if action == 2:
        attLoc1 = i
        attLoc2 = j - 1

    if action == 3:
        attLoc1 = i
        attLoc2 = j + 1

    if world[attLoc1, attLoc2, 0].passable == 1:
        if world[attLoc1, attLoc2, 0].appearence == [0.0, 0.0, 255.0]:
            reward = 10
            wolfEats = wolfEats + 1
        world[i, j, 0] = EmptyObject()
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        reward = 0
    else:
        if world[attLoc1, attLoc2, 0].kind == "wall":
            reward = -0.1
        if world[attLoc1, attLoc2, 0].kind == "agent":
            reward = 10
            gamePoints[1] = gamePoints[1] + 1
            lastexp = world[attLoc1, attLoc2, 0].replay[-1]
            # need to ensure that the agent knows that it is dying
            exp = (lastexp[0], lastexp[1], -25, lastexp[3], 1)
            # world[attLoc1, attLoc2, 0].reward -= 25
            world[attLoc1, attLoc2, 0].died()
            world[attLoc1, attLoc2, 0].replay.append(exp)

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[newLoc1, newLoc2, 0].replay.append(exp)
        world[newLoc1, newLoc2, 0].reward += reward

    return world, models, gamePoints
