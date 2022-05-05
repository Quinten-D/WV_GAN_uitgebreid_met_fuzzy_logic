import math

import torch


class NotF(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return 1. - x

    @staticmethod
    def backward(self, grad_output):
        return -1. * grad_output


class AndF(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, p):
        """###
        x = x.double()
        y = y.double()
        ###"""
        self.save_for_backward(x, y)
        self.p = p
        if x == 0. and y == 0. and p == 0.:
            return torch.zeros(1)
        #print("AndF forward: ", (x * y) / ((p + (1. - p) * (x + y - x * y)) ** 2))   # noemer gekwardateerd
        return (x * y) / (p + (1. - p) * (x + y - x * y))
        #noemer = torch.clamp((p + (1. - p) * (x + y - x * y)), min=math.e ** -90.)
        #return (x * y) / noemer

    @staticmethod
    def backward(self, grad_output):
        x, y = self.saved_tensors
        if x == 0 and y == 0 and self.p == 0:
            return torch.zeros(1), torch.zeros(1), None
        noemer = torch.clamp((self.p + (1. - self.p) * (x + y - x * y)) ** 2, min=math.e ** -90.)
        x_grad = grad_output * ((y * self.p + (1. - self.p) * (y ** 2)) / noemer)
        y_grad = grad_output * ((x * self.p + (1. - self.p) * (x ** 2)) / noemer)
        #print("AndF backward x_grad: ", x_grad)
        return x_grad, y_grad, None


class ImpF(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, p):
        self.save_for_backward(x, y)
        self.p = p
        if x <= y:
            return torch.ones(1)
        return (x * y + p * y - p * x * y) / (x - y + x * y + p * y - p * x * y)

    @staticmethod
    def backward(self, grad_output):
        #print("grad output: ", grad_output)
        x, y = self.saved_tensors
        if x <= y:
            return torch.zeros(1), torch.zeros(1), None
        noemer = torch.clamp(((x - y + x * y + self.p * y - self.p * x * y) ** 2), min=math.e ** -90.)
        x_grad = grad_output * ((self.p * (y ** 2) - self.p * y - (y ** 2)) /
                                noemer)
        y_grad = grad_output * (((x ** 2) + self.p * x - self.p * (x ** 2)) /
                                noemer)
        #print("IMP backward: ", x_grad)
        return x_grad, y_grad, None


class Log(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        self.limit = math.e ** -100.
        if x <= self.limit:
            return torch.tensor([-100.])
        return torch.tensor([math.log(x)])

    @staticmethod
    def backward(self, grad_output):
        x, = self.saved_tensors
        if x <= self.limit:
            return grad_output*(1/self.limit)
        return grad_output*(1/x)


def or_f(x, y, p):
    not_f = NotF.apply
    and_f = AndF.apply
    return not_f(and_f(not_f(x), not_f(y), p))


def xor_f(x, y, p):
    not_f = NotF.apply
    and_f = AndF.apply
    return and_f(or_f(x, y, p), not_f(and_f(x, y, p)), p)


def is_f(maze, row, column, tile):
    return maze[4 * column + 31 * 4 * row + tile]


def constraint1(maze, p):
    and_f = AndF.apply
    result = and_f(is_f(maze, 0, 1, 2), is_f(maze, 30, 29, 3), p)
    i = 0
    while i < 31:
        result = and_f(result, is_f(maze, i, 0, 1), p)
        result = and_f(result, is_f(maze, i, 30, 1), p)
        i += 1
    i = 0
    while i < 28:
        result = and_f(result, is_f(maze, 0, i + 2, 1), p)
        result = and_f(result, is_f(maze, 30, i, 1), p)
        i += 1
    return result


def constraint2(maze, p):
    and_f = AndF.apply
    result = and_f(is_f(maze, 1, 1, 0), is_f(maze, 29, 29, 0), p)
    i = 1
    while i < 29:
        result = and_f(result, or_f(is_f(maze, 1, i+1, 0), is_f(maze, 1, i+1, 1), p), p)
        result = and_f(result, or_f(is_f(maze, 29, i, 0), is_f(maze, 29, i, 1), p), p)
        i += 1
    i = 2
    while i < 29:
        j = 1
        while j < 30:
            result = and_f(result, or_f(is_f(maze, i, j, 0), is_f(maze, i, j, 1), p), p)
            j += 1
        i += 1
    return result


def constraint21(maze, p):
    and_f = AndF.apply
    imp_f = ImpF.apply
    result = is_f(maze, 29, 29, 0)
    i = 1
    while i < 29:
        j = 1
        while j < 29:
            result = and_f(result, or_f(is_f(maze, i, j, 0), is_f(maze, i, j, 1), p), p)
            result = and_f(result, imp_f(and_f(is_f(maze, i, j, 1), is_f(maze, i + 1, j + 1, 1), p),
                                         xor_f(is_f(maze, i, j + 1, 1), is_f(maze, i + 1, j, 1), p), p), p)
            result = and_f(result, imp_f(and_f(is_f(maze, i + 1, j, 1), is_f(maze, i, j + 1, 1), p),
                                         xor_f(is_f(maze, i, j, 1), is_f(maze, i + 1, j + 1, 1), p), p), p)
            j += 1
        result = and_f(result, or_f(is_f(maze, i, 29, 0), is_f(maze, i, 29, 1), p), p)
        result = and_f(result, or_f(is_f(maze, 29, i, 0), is_f(maze, 29, i, 1), p), p)
        i += 1
    ##
    #clamped_value = torch.clamp(result, min=math.e ** -90.)  # 90 voor enkel constraint 1
    #result = clamped_value
    ##
    return result


def all_constraints(mazes, p):
    and_f = AndF.apply
    #log = Log.apply
    batch_size = list(mazes.size())[0]
    result = torch.empty(batch_size, requires_grad=False)
    i = 0
    #print("--- resultaten fuzzy logic functie ---")
    while i < batch_size:
        #print(constraint1(mazes[i, :], p))
        #result[i] = -1*log(constraint1(mazes[i, :], p))
        #result[i] = -1 * constraint1(mazes[i, :], p)   #enkel log weggehaald
        # result[i] = -1 * torch.log(constraint1(mazes[i, :], p))   # torch log
        #fuzzy_formula_output = and_f(constraint1(mazes[i, :], p), constraint2(mazes[i, :], p), p)
        fuzzy_formula_output = and_f(constraint1(mazes[i, :], p), constraint2(mazes[i, :], p), p)
        #fuzzy_formula_output = torch.add(fuzzy_formula_output, math.e ** -95.)
        #print(fuzzy_formula_output)
        clamped_value = torch.clamp(fuzzy_formula_output, min=math.e ** -90.)   # 90 voor enkel constraint 1
        #clamped_value = fuzzy_formula_output
        #print("clamp: ", clamped_value)
        result[i] = -1 * torch.log(clamped_value)   # clamp   log ontbreekt hier

        i += 1
    #print("------")

    #print("gemiddelde loss van fuzzy logic term: ", torch.mean(result))
    return torch.mean(result)


def some_constraints(mazes, p):
    and_f = AndF.apply
    #log = Log.apply
    batch_size = list(mazes.size())[0]
    result = torch.empty(batch_size, requires_grad=False)
    i = 0
    #print("--- resultaten fuzzy logic functie ---")
    while i < batch_size:
        #print(constraint1(mazes[i, :], p))
        #result[i] = -1*log(constraint1(mazes[i, :], p))
        #result[i] = -1 * constraint1(mazes[i, :], p)   #enkel log weggehaald
        # result[i] = -1 * torch.log(constraint1(mazes[i, :], p))   # torch log
        #fuzzy_formula_output = and_f(constraint1(mazes[i, :], p), constraint2(mazes[i, :], p), p)
        fuzzy_formula_output = constraint1(mazes[i, :], p)
        #fuzzy_formula_output = torch.add(fuzzy_formula_output, math.e ** -95.)
        #print(fuzzy_formula_output)
        clamped_value = torch.clamp(fuzzy_formula_output, min=math.e ** -90.)   # 90 voor enkel constraint 1
        #clamped_value = fuzzy_formula_output
        #print("clamp: ", clamped_value)
        result[i] = -1 * torch.log(clamped_value)   # clamp   log ontbreekt hier

        i += 1
    #print("------")

    #print("gemiddelde loss van fuzzy logic term: ", torch.mean(result))
    return torch.mean(result)


class FuzzyLogic(torch.nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, maze):
        maze = maze.double()   # aub zet deze tensor om in double
        result = all_constraints(maze, self.p)
        return result.float()


class FuzzyLogic2(torch.nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, maze):
        maze = maze.double()   # aub zet deze tensor om in double
        result = some_constraints(maze, self.p)
        return result.float()


def test_constraint1(sample, p):
    batch_size = list(sample.size())[0]
    result = torch.empty(batch_size, requires_grad=False)
    i = 0
    while i < batch_size:
        result[i] = constraint1(sample[i, :], p)
        i += 1
    return torch.mean(result).float()


def test_constraint2(sample, p):
    batch_size = list(sample.size())[0]
    result = torch.empty(batch_size, requires_grad=False)
    i = 0
    while i < batch_size:
        result[i] = constraint2(sample[i, :], p)
        i += 1
    return torch.mean(result).float()


def test_all_constraint(sample, p):
    and_f = AndF.apply
    batch_size = list(sample.size())[0]
    result = torch.empty(batch_size, requires_grad=False)
    i = 0
    while i < batch_size:
        result[i] = and_f(constraint1(sample[i, :], p), constraint2(sample[i, :], p), p)
        i += 1
    return torch.mean(result).float()


x = torch.rand(1, dtype=torch.double, requires_grad=True)
y = torch.rand(1, dtype=torch.double, requires_grad=True)
# print(x)
# print(y)
torch.autograd.gradcheck(NotF.apply, x)
torch.autograd.gradcheck(AndF.apply, (x, y, 0))
torch.autograd.gradcheck(ImpF.apply, (x, y, 0))


