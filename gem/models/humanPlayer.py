class modelClassPlayer:

    kind = "humanPlayer"  # class variable shared by all instances

    def __init__(self, actionSpace, replaySize):
        self.modeltype = "humanPlayer"
        self.actionSpace = actionSpace
        self.inputType = "keyboard"
        self.replay = deque([], maxlen=replaySize)

    def takeAction(self, params):
        pytorchInput, epsilon = params

        # img = needs to convert the pytorchInput back into RGB
        # change (1,3,9,9) back to (9,9,3)

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

        done = 0
        while done == 0:
            action = int(input("Select Action: "))
            if action in self.actionSpace:
                done = 1
            else:
                print("Please try again. Possible actions are below.")
                print(self.actionSpace)
                # we can have iinputType above also be joystick, or other controller

        return action
