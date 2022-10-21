import math
from enum import Enum
import random

import pygame
import pygame as pg
import sys


class PointType(Enum):
    DEFAULT = 0
    ROOT = 1
    BORDER = 2
    NOISE = 3

class Point:
    def __init__(self, x ,y, point_type=PointType.DEFAULT, color='black', is_visited = False):
        self.x = x
        self.y = y
        self.point_type = point_type
        self.color = color
        self.is_visited = is_visited


sc = pg.display.set_mode((800, 600))
sc.fill('white')
pg.display.update()

list_of_points = []


def is_point_already_exist(points, new_point):
    for point in points:
        if point.x == new_point.x and point.y == new_point.y:
            return True

    return False


def get_points_neighbours(point, points, max_distance):
    neighbours = []
    for current_point in points:
        if point == current_point:
            continue
        else:
            distance = get_distance(current_point, point)
            if distance <= max_distance:
                neighbours.append(current_point)

    return neighbours


def get_groups(points, max_distance, min_neighbours):
    groups = []

    for point in points:
        if point.is_visited:
            continue
        else:
            neighbours = get_points_neighbours(point, points, max_distance)
            if len(neighbours) >= min_neighbours:
                point.is_visited = True
                point.point_type = PointType.ROOT
                groups.append([point])
                while len(neighbours) > 0:
                    current_neighbour = neighbours.pop(0)
                    if current_neighbour.is_visited:
                        continue
                    sub_neighbours = get_points_neighbours(current_neighbour, points, max_distance)
                    if len(sub_neighbours) < min_neighbours or current_neighbour.point_type == PointType.NOISE:
                        current_neighbour.is_visited = True
                        current_neighbour.point_type = PointType.BORDER
                        groups[len(groups)-1].append(current_neighbour)
                    else:
                        current_neighbour.is_visited = True
                        current_neighbour.point_type = PointType.ROOT
                        groups[len(groups) - 1].append(current_neighbour)
                        neighbours.extend(sub_neighbours)
            else:
                point.point_type = PointType.NOISE

    return groups


def get_distance(first_point, second_point):
    return math.sqrt(math.pow(first_point.x - second_point.x, 2) + math.pow(first_point.y - second_point.y, 2))


def refresh_points(points):
    for point in points:
        point.is_visited = False
        point.point_type = PointType.DEFAULT


def colorize_points_by_flags(points):
    sc.fill('white')
    pg.display.update()

    for point in points:
        color = 'black'
        if point.point_type == PointType.NOISE:
            color = 'red'
        elif point.point_type == PointType.BORDER:
            color = 'yellow'
        elif point.point_type == PointType.ROOT:
            color = 'green'
        else:
            color = 'black'
        pg.draw.circle(sc, color, (point.x, point.y), 5)
        pg.display.update()


def colorize_points_by_groups(groups, points):
    sc.fill('white')
    pg.display.update()
    for list in groups:
        random_color = "#%06x" % random.randint(0, 0xFFFFFF)
        for item in list:
            pg.draw.circle(sc, random_color, (item.x, item.y), 5)
            pg.display.update()

    for point in points:
        if point.point_type == PointType.NOISE:
            pg.draw.circle(sc, 'red', (point.x, point.y), 5)
            pg.display.update()


while True:
    for i in pg.event.get():
        if i.type == pg.QUIT:
            sys.exit()

        if i.type == pygame.KEYDOWN:
            if i.key == pygame.K_RETURN:
                refresh_points(list_of_points)
                groups = get_groups(list_of_points, 20, 3)
                colorize_points_by_flags(list_of_points)
            if i.key == pygame.K_ESCAPE:
                refresh_points(list_of_points)
                groups = get_groups(list_of_points, 20, 3)
                colorize_points_by_groups(groups, list_of_points)

    pressed = pg.mouse.get_pressed()
    pos = pg.mouse.get_pos()
    if pressed[0]:
        point = Point(pos[0], pos[1])
        if not is_point_already_exist(list_of_points, point):
            list_of_points.append(point)
            pg.draw.circle(sc, point.color, pos, 5)
            pg.display.update()

    pg.time.delay(100)

