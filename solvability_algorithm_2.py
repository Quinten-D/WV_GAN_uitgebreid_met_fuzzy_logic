import queue
from train_GAN_4_fl import *


def is_f(maze, row, column, tile):
    return maze[4 * column + 31 * 4 * row + tile] == 1


def data(row, column):
    return 30 - row + 29 - column, (row, column)


def neighbour(x, y):
    neighbours = []
    if x != 0:
        neighbours.append((x-1, y))
    if x != 30:
        neighbours.append((x+1, y))
    if y != 0:
        neighbours.append((x, y-1))
    if y != 30:
        neighbours.append((x, y+1))
    return neighbours


def solvable(maze):
    open_set = queue.PriorityQueue()
    reachable_tiles = []
    # check if entry tile is correct
    if is_f(maze, 0, 1, 2):
        open_set.put(data(0, 1))
        reachable_tiles.append((0,1))

    closed_set = []

    while not open_set.empty():
        position = open_set.get()[1]
        closed_set.append(position)
        neighbours = neighbour(position[0], position[1])
        for pos in neighbours:
            if is_f(maze, pos[0], pos[1], 0) and pos not in closed_set:
                open_set.put(data(pos[0], pos[1]))
                reachable_tiles.append(pos)
            elif is_f(maze, pos[0], pos[1], 3) and pos == (30, 29) and pos not in closed_set:
                open_set.put(data(pos[0], pos[1]))
                reachable_tiles.append(pos)

    # turn reachable_tiles into reachability map
    reachability_map = [0] * 961
    for pos in reachable_tiles:
        row = pos[0]
        column = pos[1]
        reachability_map[column + 31 * row] = 1
    return reachability_map


if __name__=='__main__':
    # om resultaten te plotten
    def plot_img2(array, number=None):
        array = array.reshape(31, 31)
        print(array)
        plt.imshow(array, cmap='binary')
        plt.xticks([])
        plt.yticks([])
        if number:
            plt.xlabel(number, fontsize='x-large')
        plt.show()

    dataset = MyDataset('levels_corrected.csv')
    loader = DataLoader(
        dataset,
        batch_size=40,
        shuffle=True,
        num_workers=2
    )

    for d in loader:
        break
    #print(d[0])
    maze = d[0]
    plot_img(maze)

    print("maze solvable: ", solvable(maze))
    t = torch.tensor(solvable(maze), dtype=torch.float32)
    plot_img2(t)
    #m = solvable(maze)
    #print(torch.tensor(solvable(m)).size())
    #print(t)
    #print(t[4])


