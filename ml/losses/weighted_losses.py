import torch


class WeightedBCELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input1, input2, weights):
        out = torch.sum(input1 * input2, -1)

        return self.bce_loss(out, weights)


class WeightedL1Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.goal = 10
        self.goal_increaser = lambda x: x+0.002

    def forward(self, input1, input2, weights):
        out = torch.sum(input1 * input2, -1)

        self.goal = self.goal_increaser(self.goal)

        weights *= self.goal
        weights[weights == 0] = -self.goal

        return self.l1_loss(out, weights)


class WeightedL2Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL2Loss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, input1, input2, weights):
        out = torch.sum(input1 * input2, -1)
        weights *= 10
        weights[weights == 0] = -10

        return self.l2_loss(out, weights)
