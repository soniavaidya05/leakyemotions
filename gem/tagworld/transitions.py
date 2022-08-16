from gem.environment.elements import EmptyObject, tagAgent, Wall
from models.perception import agentVisualField
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
#                      Tag Agent transition rules
# ---------------------------------------------------------------------

def tagAgentTransitions(holdObject, action, world, models, i, j, rewards, totalRewards, done, input, expBuff=True):

    newLoc1 = i
    newLoc2 = j

    reward = 0

    # if object is frozen, just decrease frozen count and then go from there
    # hope this doesn't break learning
    if holdObject.frozen > 0:
        holdObject.frozen -= 1
        return world, models, totalRewards

    if action == 0:
        attLoc1 = i-1
        attLoc2 = j
    elif action == 1:
        attLoc1 = i+1
        attLoc2 = j
    elif action == 2:
        attLoc1 = i
        attLoc2 = j-1
    elif action == 3:
        attLoc1 = i
        attLoc2 = j+1

    if world[attLoc1, attLoc2, 0].passable == 1:
        world[i, j, 0] = EmptyObject()
        reward += 1
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        totalRewards += 1
    elif world[attLoc1, attLoc2, 0].kind == "tagAgent":
        # bump into each other
        alterAgent = world[attLoc1, attLoc2, 0]
        if holdObject.is_it == 1 & alterAgent.is_it == 0:
            reward += 7
            holdObject.tag()  # change who is it
            alterAgent.tag()

            lastexp = world[attLoc1, attLoc2, 0].replay[-1]
            exp = (lastexp[0], lastexp[1], -15, lastexp[3], 1)
            world[attLoc1, attLoc2, 0].replay.append(exp)

        elif holdObject.is_it == 0 & alterAgent.is_it == 1:
            # agent bumpbed into tag person
            reward -= 13
            holdObject.tag()  # change who is it
            alterAgent.tag()

        elif holdObject.is_it == alterAgent.is_it:
            # bump into each other
            reward -= .1

    elif world[attLoc1, attLoc2, 0].kind == "wall":
        reward = -.1

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[newLoc1, newLoc2, 0].replay.append(exp)
        world[newLoc1, newLoc2, 0].reward += reward

    return world, models, totalRewards


def tagAgentTransitionsOld(holdObject, action, world, models, i, j, rewards, totalRewards, done, input, expBuff=True):

    newLoc1 = i
    newLoc2 = j

    reward = 0

    # if object is frozen, just decrease frozen count and then go from there
    # hope this doesn't break learning
    if holdObject.frozen > 0:
        holdObject.frozen -= 1
        return world, models, totalRewards

    if action == 0:
        attLoc1 = i-1
        attLoc2 = j
    elif action == 1:
        attLoc1 = i+1
        attLoc2 = j
    elif action == 2:
        attLoc1 = i
        attLoc2 = j-1
    elif action == 3:
        attLoc1 = i
        attLoc2 = j+1

    if world[attLoc1, attLoc2, 0].passable == 1:
        world[i, j, 0] = EmptyObject()
        reward += 1
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        totalRewards += 1
    elif world[attLoc1, attLoc2, 0].kind == "tagAgent":
        # bump into each other
        alterAgent = world[attLoc1, attLoc2, 0]
        if holdObject.is_it == 1 & alterAgent.is_it == 0:
            reward += 7
            holdObject.tag()  # change who is it
            alterAgent.tag()

            if expBuff == True:

                imgDead = agentVisualField(
                    world, (attLoc1, attLoc2), alterAgent.vision)
                inputDead = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                QDead = models[world[attLoc1, attLoc2, 0].policy].model1(
                    inputDead)
                pDead = m(QDead).detach().numpy()[0]
                actionDead = np.random.choice([0, 1, 2, 3], p=pDead)

                imgDead2 = agentVisualField(
                    world, (attLoc1, attLoc2, j), world[attLoc1, attLoc2, 0].vision)
                inputDead2 = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                exp = (inputDead, actionDead, -10, inputDead2, 1)
                models[world[attLoc1, attLoc2, 0].policy].replay.append(exp)
        elif holdObject.is_it == 0 & alterAgent.is_it == 1:
            # agent bumpbed into tag person
            reward -= 13
            holdObject.tag()  # change who is it
            alterAgent.tag()

            if expBuff == True:

                imgDead = agentVisualField(
                    world, (attLoc1, attLoc2), alterAgent.vision)
                inputDead = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                QDead = models[world[attLoc1, attLoc2, 0].policy].model1(
                    inputDead)
                pDead = m(QDead).detach().numpy()[0]
                actionDead = np.random.choice([0, 1, 2, 3], p=pDead)

                imgDead2 = agentVisualField(
                    world, (attLoc1, attLoc2, j), world[attLoc1, attLoc2, 0].vision)
                inputDead2 = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                exp = (inputDead, actionDead, 7, inputDead2, 1)
                models[world[attLoc1, attLoc2, 0].policy].replay.append(exp)
        elif holdObject.is_it == alterAgent.is_it:
            # bump into each other
            reward -= .1

    elif world[attLoc1, attLoc2, 0].kind == "wall":
        reward = -.1

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        models[holdObject.policy].replay.append(exp)

    return world, models, totalRewards



def agentTransitions(
    holdObject, action, world, models, i, j, totalRewards, done, input, expBuff=True
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
        totalRewards = totalRewards + reward
    else:
        if world[attLoc1, attLoc2, 0].kind == "wall":
            reward = -0.1

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[newLoc1, newLoc2, 0].replay.append(exp)
        world[newLoc1, newLoc2, 0].reward = +reward

    return world, models, totalRewards


# ---------------------------------------------------------------------
#                      Wolf transition rules
# ---------------------------------------------------------------------


def wolfTransitions(
    holdObject, action, world, models, i, j, wolfEats, done, input, expBuff=True
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
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        reward = 0
    else:
        if world[attLoc1, attLoc2, 0].kind == "wall":
            reward = -0.1
        if world[attLoc1, attLoc2, 0].kind == "agent":
            reward = 10
            lastexp = world[attLoc1, attLoc2, 0].replay[-1]
            # need to ensure that the agent knows that it is dying
            exp = (lastexp[0], lastexp[1], -15, lastexp[3], 1)
            world[attLoc1, attLoc2, 0].replay.append(lastexp)

    if expBuff == True:

        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        world[newLoc1, newLoc2, 0].replay.append(exp)
        world[newLoc1, newLoc2, 0].reward = +reward

    return world, models, wolfEats
