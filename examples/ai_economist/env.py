from examples.ai_economist.elements import (
    Agent,
    Wood,
    Stone,
    House,
    EmptyObject,
    Wall,
)
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception import agent_visualfield




class AI_Econ:
    def __init__(
        self,
        height=30,
        width=30,
        layers=2,
        defaultObject=EmptyObject(),
        wood1p=0.04,
        stone1p=0.04,
    ):
        self.wood1p = wood1p
        self.stone1p = stone1p
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.wood1p, self.stone1p)
        self.insert_walls(self.height, self.width, self.layers)
        self.wood = 4
        self.stone = 4

    def create_world(self, height=30, width=30, layers=2):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(self, height=30, width=30, layers=1, wood1p=0.04, stone1p=0.04):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(wood1p, stone1p)
        self.insert_walls(height, width, layers)

    def plot(self, layer):  # is this defined in the master?
        """
        Creates an RGB image of the whole world
        """
        image_r = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_g = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_b = np.random.random((self.world.shape[0], self.world.shape[1]))

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                image_r[i, j] = self.world[i, j, layer].appearance[0]
                image_g[i, j] = self.world[i, j, layer].appearance[1]
                image_b[i, j] = self.world[i, j, layer].appearance[2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image

    def init_elements(self):
        """
        Creates objects that survive from game to game
        """
        self.emptyObject = EmptyObject()
        self.walls = Wall()

    def game_test(self, layer=0):
        """
        Prints one frame to check game instance parameters
        """
        image = self.plot(layer)

        moveList = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i, j, layer].static == 0:
                    moveList.append([i, j, layer])

        if len(moveList) > 0:
            img = agent_visualfield(self.world, moveList[0], k=4)
        else:
            img = image

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self, wood1p=0.04, stone1p=0.04):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2],
                    p=[
                        wood1p,
                        stone1p,
                        1 - wood1p - stone1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = Wood()
                if obj == 1:
                    self.world[i, j, 0] = Stone()

        loc = (3, 7, 1)
        apperence1 = (0., 0., 255.0)
        apperence2 = (50., 0., 255.0)
        apperence3 = (0., 50., 255.0)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (3, 4, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (7, 4, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (23, 7, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (23, 4, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (27, 4, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (23, 27, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (23, 24, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (27, 24, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (3, 23, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (3, 27, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (7, 24, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

    def insert_walls(self, height, width, layers):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        for layer in range(layers):

            for i in range(height):
                self.world[0, i, layer] = Wall()
                self.world[height - 1, i, layer] = Wall()
                self.world[i, 0, layer] = Wall()
                self.world[i, height - 1, layer] = Wall()

            # this is a hack to get to look like AI economist
            for i in range(8):
                self.world[14, i, layer] = Wall()
                self.world[i, 14, layer] = Wall()
            for i in range(8):
                self.world[14, height - i - 1, layer] = Wall()
                self.world[height - i - 1, 14, layer] = Wall()

    def step(self, models, loc, epsilon=0.85):
        """
        This is an example script for an alternative step function
        It does not account for the fact that an agent can die before
        it's next turn in the moveList. If that can be solved, this
        may be preferable to the above function as it is more like openAI gym

        The solution may come from the agent.died() function if we can get that to work

        location = (i, j, 0)

        Uasge:
            for i, j, k = agents
                location = (i, j, k)
                state, action, reward, next_state, done, additional_output = env.stepSingle(models, (0, 0, 0), epsilon)
                env.world[0, 0, 0].updateMemory(state, action, reward, next_state, done, additional_output)
            env.WorldUpdate()

        """
        holdObject = self.world[loc]
        device = models[holdObject.policy].device

        if holdObject.static != 1:
            """
            This is where the agent will make a decision
            If done this way, the pov statement may be about to be part of the action
            Since they are both part of the same class

            if going for this, the pov statement needs to know about location rather than separate
            i and j variables
            """
            state = models[holdObject.policy].pov(
                self.world,
                loc,
                holdObject,
                inventory=[holdObject.stone, holdObject.wood, holdObject.coin],
                layers=[0, 1],
            )
            action = models[holdObject.policy].take_action([state.to(device), epsilon])

        if holdObject.has_transitions == True:
            """
            Updates the world given an action
            TODO: does this need self.world in here, or can it be figured out by passing self?
            """
            (
                self.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = holdObject.transition(self, models, action, loc)
        else:
            reward = 0
            next_state = state

        additional_output = []

        return state, action, reward, next_state, done, new_loc, additional_output
