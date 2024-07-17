
# import math
# import random
# import matplotlib.pyplot as plt
# import sys
# import pathlib
# sys.path.append(str(pathlib.Path(__file__).parent))

# from rrt import RRT

# show_animation = True


# def get_path_length(path):
#     le = 0
#     for i in range(len(path) - 1):
#         dx = path[i + 1][0] - path[i][0]
#         dy = path[i + 1][1] - path[i][1]
#         d = math.hypot(dx, dy)
#         le += d

#     return le


# def get_target_point(path, targetL):
#     le = 0
#     ti = 0
#     lastPairLen = 0
#     for i in range(len(path) - 1):
#         dx = path[i + 1][0] - path[i][0]
#         dy = path[i + 1][1] - path[i][1]
#         d = math.hypot(dx, dy)
#         le += d
#         if le >= targetL:
#             ti = i - 1
#             lastPairLen = d
#             break

#     partRatio = (le - targetL) / lastPairLen

#     x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
#     y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

#     return [x, y, ti]


# def line_collision_check(first, second, obstacleList):
#     # Line Equation

#     x1 = first[0]
#     y1 = first[1]
#     x2 = second[0]
#     y2 = second[1]

#     try:
#         a = y2 - y1
#         b = -(x2 - x1)
#         c = y2 * (x2 - x1) - x2 * (y2 - y1)
#     except ZeroDivisionError:
#         return False

#     for (ox, oy, size) in obstacleList:
#         d = abs(a * ox + b * oy + c) / (math.hypot(a, b))
#         if d <= size:
#             return False

#     return True  # OK


# def path_smoothing(path, max_iter, obstacle_list):
#     le = get_path_length(path)

#     for i in range(max_iter):
#         # Sample two points
#         pickPoints = [random.uniform(0, le), random.uniform(0, le)]
#         pickPoints.sort()
#         first = get_target_point(path, pickPoints[0])
#         second = get_target_point(path, pickPoints[1])

#         if first[2] <= 0 or second[2] <= 0:
#             continue

#         if (second[2] + 1) > len(path):
#             continue

#         if second[2] == first[2]:
#             continue

#         # collision check
#         if not line_collision_check(first, second, obstacle_list):
#             continue

#         # Create New path
#         newPath = []
#         newPath.extend(path[:first[2] + 1])
#         newPath.append([first[0], first[1]])
#         newPath.append([second[0], second[1]])
#         newPath.extend(path[second[2] + 1:])
#         path = newPath
#         le = get_path_length(path)

#     return path


# def main():
#     # ====Search Path with RRT====
#     # Parameter
#     obstacleList = [
#         (5, 5, 1),
#         (3, 6, 2),
#         (3, 8, 2),
#         (3, 10, 2),
#         (7, 5, 2),
#         (9, 5, 2)
#     ]  # [x,y,size]
#     rrt = RRT(start=[0, 0], goal=[6, 10],
#               rand_area=[-2, 15], obstacle_list=obstacleList)
#     path = rrt.planning(animation=show_animation)

#     # Path smoothing
#     maxIter = 1000
#     smoothedPath = path_smoothing(path, maxIter, obstacleList)

#     # Draw final path
#     if show_animation:
#         rrt.draw_graph()
#         plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

#         plt.plot([x for (x, y) in smoothedPath], [
#             y for (x, y) in smoothedPath], '-c')

#         plt.grid(True)
#         plt.pause(0.1)  # Need for Mac
#         plt.show()


# if __name__ == '__main__':
#     main()


import math
import random
import numpy as np
from examples.rrt import RRT

def get_path_length(path):
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        dz = path[i + 1][2] - path[i][2]
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        le += d

    return le


def get_target_point(path, targetL):
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        dz = path[i + 1][2] - path[i][2]
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break

    if lastPairLen == 0:
        lastPairLen = 1e-6  # small non-zero value to avoid division by zero

    partRatio = (le - targetL) / lastPairLen

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio
    z = path[ti][2] + (path[ti + 1][2] - path[ti][2]) * partRatio

    return [x, y, z, ti]



def line_collision_check(first, second, obstacleList):
    x1 = first[0]
    y1 = first[1]
    z1 = first[2]
    x2 = second[0]
    y2 = second[1]
    z2 = second[2]

    for (ox, oy, oz, size) in obstacleList:
        d = np.linalg.norm(np.cross(np.array([x2-x1, y2-y1, z2-z1]), np.array([x1-ox, y1-oy, z1-oz]))) / np.linalg.norm(np.array([x2-x1, y2-y1, z2-z1]))
        if d <= size:
            return False

    return True  # OK


def path_smoothing(path, max_iter, obstacle_list):
    le = get_path_length(path)

    for i in range(max_iter):
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])

        if first[3] <= 0 or second[3] <= 0:
            continue

        if (second[3] + 1) > len(path):
            continue

        if second[3] == first[3]:
            continue

        if not line_collision_check(first, second, obstacle_list):
            continue

        newPath = []
        newPath.extend(path[:first[3] + 1])
        newPath.append([first[0], first[1], first[2]])
        newPath.append([second[0], second[1], second[2]])
        newPath.extend(path[second[3] + 1:])
        path = newPath
        le = get_path_length(path)

    return path


class SmoothRRT:
    def __init__(self, start, goal, steps, obstacle_list):
        self.start = start
        self.goal = goal
        self.steps = steps
        self.obstacle_list = obstacle_list

    def generate_path(self):
        rrt = RRT(start=self.start, goal=self.goal, rand_area=[-2, 15], obstacle_list=self.obstacle_list)
        path = rrt.planning(animation=False)
        if path is None:
            raise ValueError("Could not find a path with RRT")
        
        smoothed_path = path_smoothing(path, 1000, self.obstacle_list)
        return np.linspace(self.start, self.goal, self.steps).tolist()

def generate_smooth_rrt(start, goal, steps, obstacle_list):
    rrt = SmoothRRT(start, goal, steps, obstacle_list)
    return rrt.generate_path()
