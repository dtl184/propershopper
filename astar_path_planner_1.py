
import numpy as np
from utils import recv_socket_data
import json
from queue import PriorityQueue

import json
import socket

from utils import recv_socket_data

objs = [
    {'height': 2.5, 'width': 3, 'position': [0.2, 4.5], 're_centered_position': [2.125, 5.75]},
    {'height': 2.5, 'width': 3, 'position': [0.2, 9.5], 're_centered_position': [2.125, 10.75]},
    {'height': 1, 'width': 2, 'position': [5.5, 1.5], 're_centered_position': [6.5, 2]},
    {'height': 1, 'width': 2, 'position': [7.5, 1.5], 're_centered_position': [8.5, 2]},
    {'height': 1, 'width': 2, 'position': [9.5, 1.5], 're_centered_position': [10.5, 2]},
    {'height': 1, 'width': 2, 'position': [11.5, 1.5], 're_centered_position': [12.5, 2]},
    {'height': 1, 'width': 2, 'position': [13.5, 1.5], 're_centered_position': [14.5, 2]},
    {'height': 1, 'width': 2, 'position': [5.5, 5.5], 're_centered_position': [6.5, 6]},
    {'height': 1, 'width': 2, 'position': [7.5, 5.5], 're_centered_position': [8.5, 6]},
    {'height': 1, 'width': 2, 'position': [9.5, 5.5], 're_centered_position': [10.5, 6]},
    {'height': 1, 'width': 2, 'position': [11.5, 5.5], 're_centered_position': [12.5, 6]},
    {'height': 1, 'width': 2, 'position': [13.5, 5.5], 're_centered_position': [14.5, 6]},
    {'height': 1, 'width': 2, 'position': [5.5, 9.5], 're_centered_position': [6.5, 10]},
    {'height': 1, 'width': 2, 'position': [7.5, 9.5], 're_centered_position': [8.5, 10]},
    {'height': 1, 'width': 2, 'position': [9.5, 9.5], 're_centered_position': [10.5, 10]},
    {'height': 1, 'width': 2, 'position': [11.5, 9.5], 're_centered_position': [12.5, 10]},
    {'height': 1, 'width': 2, 'position': [13.5, 9.5], 're_centered_position': [14.5, 10]},
    {'height': 1, 'width': 2, 'position': [5.5, 13.5], 're_centered_position': [6.5, 14]},
    {'height': 1, 'width': 2, 'position': [7.5, 13.5], 're_centered_position': [8.5, 14]},
    {'height': 1, 'width': 2, 'position': [9.5, 13.5], 're_centered_position': [10.5, 14]},
    {'height': 1, 'width': 2, 'position': [11.5, 13.5], 're_centered_position': [12.5, 14]},
    {'height': 1, 'width': 2, 'position': [13.5, 13.5], 're_centered_position': [14.5, 14]},
    {'height': 1, 'width': 2, 'position': [5.5, 17.5], 're_centered_position': [6.5, 18]},
    {'height': 1, 'width': 2, 'position': [7.5, 17.5], 're_centered_position': [8.5, 18]},
    {'height': 1, 'width': 2, 'position': [9.5, 17.5], 're_centered_position': [10.5, 18]},
    {'height': 1, 'width': 2, 'position': [11.5, 17.5], 're_centered_position': [12.5, 18]},
    {'height': 1, 'width': 2, 'position': [13.5, 17.5], 're_centered_position': [14.5, 18]},
    {'height': 1, 'width': 2, 'position': [5.5, 21.5], 're_centered_position': [6.5, 22]},
    {'height': 1, 'width': 2, 'position': [7.5, 21.5], 're_centered_position': [8.5, 22]},
    {'height': 1, 'width': 2, 'position': [9.5, 21.5], 're_centered_position': [10.5, 22]},
    {'height': 1, 'width': 2, 'position': [11.5, 21.5], 're_centered_position': [12.5, 22]},
    {'height': 1, 'width': 2, 'position': [13.5, 21.5], 're_centered_position': [14.5, 22]},
    {'height': 6, 'width': 0.7, 'position': [1, 18.5], 're_centered_position': [1.35, 21.5]},
    {'height': 6, 'width': 0.7, 'position': [2, 18.5], 're_centered_position': [2.35, 21.5]},
    {'height': 0.8, 'width': 0.8, 'position': [3.5, 18.5], 're_centered_position': [4.15, 19.4]},
    {'height': 2.25, 'width': 1.5, 'position': [18.25, 4.75], 're_centered_position': [19.125, 5.875]},
    {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 're_centered_position': [19.125, 11.875]}
]


def update_position_to_center(obj_pose):
    """
    Update the position of objects to their re_centered_position if their current position matches obj_pose.

    Parameters:
        objects (list of dicts): List of objects with details including position and re_centered_position.
        obj_pose (list): The position to match for updating to re_centered_position.
    
    Returns:
        None: Objects are modified in place.
    """
    global objs
    for obj in objs:
        # Compare current position with obj_pose
        if obj['position'] == obj_pose:
            # If they match, update position to re_centered_position
            obj_pose = obj['re_centered_position']
            break
    return obj_pose


class Agent:
    def __init__(self, socket_game, env):
        self.shopping_list = env['observation']['players'][0]['shopping_list']
        self.shopping_quant = env['observation']['players'][0]['list_quant']
        self.game = socket_game
        self.map_width = 20
        self.map_height = 25
        self.obs = env['observation']
        self.cart = None
        self.basket = None
        self.player = self.obs['players'][0]
        self.last_action = "NOP"
        self.current_direction = self.player['direction']
        self.size = [0.6, 0.4]

        

    def step(self, action):
        #print("Sending action: ", action)
        action = "1 " + action
        self.game.send(str.encode(action))  # send action to env
        output = recv_socket_data(self.game)  # get observation from env
        #print("Observations: ", output)
        if output:
            output = json.loads(output)
            self.obs = output['observation']
            #print("Observations: ", self.obs)
            # if len(self.obs['player'][0]['carts']) > 0:
            #     self.cart = self.obs['carts'][0]
            # if len(self.obs['baskets']) > 0:
            #     self.basket = self.obs['baskets'][0]
            self.last_action = action
            self.player = self.obs['players'][0]
            #print(self.player['position'])
        return output
    
    def heuristic(self, a, b):
        """Calculate the Manhattan distance from point a to point b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def overlap(self, x1, y1, width_1, height_1, x2, y2, width_2, height_2):
        return  (x1 > x2 + width_2 or x2 > x1 + width_1 or y1 > y2 + height_2 or y2 > y1 + height_1)

    def objects_overlap(self, x1, y1, width_1, height_1, x2, y2, width_2, height_2):
        return self.overlap(x1, y1, width_1, height_1, x2, y2, width_2, height_2)
    def collision(self, x, y, width, height, obj):
        """
        Check if a rectangle defined by (x, y, width, height) does NOT intersect with an object
        and ensure the rectangle stays within the map boundaries.

        Parameters:
            x (float): The x-coordinate of the rectangle's top-left corner.
            y (float): The y-coordinate of the rectangle's top-left corner.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            obj (dict): An object with 'position', 'width', and 'height'.

        Returns:
            bool: Returns True if there is NO collision (i.e., no overlap) and the rectangle is within map boundaries,
                False if there is a collision or the rectangle goes outside the map boundaries.
        """
        # Define map boundaries
        min_x = 0.5
        max_x = 24
        min_y = 2.5
        max_y = 19.5

        # Calculate the boundaries of the rectangle
        rectangle = {
            'northmost': y,
            'southmost': y + height,
            'westmost': x,
            'eastmost': x + width
        }
        
        # Ensure the rectangle is within the map boundaries
        if not (min_x <= rectangle['westmost'] and rectangle['eastmost'] <= max_x and
                min_y <= rectangle['northmost'] and rectangle['southmost'] <= max_y):
            return False  # The rectangle is out of the map boundaries

        # Calculate the boundaries of the object
        obj_box = {
            'northmost': obj['position'][1],
            'southmost': obj['position'][1] + obj['height'],
            'westmost': obj['position'][0],
            'eastmost': obj['position'][0] + obj['width']
        }

        # Check if there is no overlap using the specified cardinal bounds
        no_overlap = not (
            (obj_box['northmost'] <= rectangle['northmost'] <= obj_box['southmost'] or
            obj_box['northmost'] <= rectangle['southmost'] <= obj_box['southmost']) and (
                (obj_box['westmost'] <= rectangle['westmost'] <= obj_box['eastmost'] or
                obj_box['westmost'] <= rectangle['eastmost'] <= obj_box['eastmost'])
            )
        )
        
        return no_overlap

    # The function will return False if the rectangle is outside the map boundaries or intersects with the object.
    

    def hits_wall(self, x, y):
        wall_width = 0.4
        return not (y <= 2 or y + self.size[1] >= self.map_height - wall_width or \
                x + self.size[0] >= self.map_width - wall_width) 
        # return y <= 2 or y + unit.height >= len(self.map) - wall_width or \
        #        x + unit.width >= len(self.map[0]) - wall_width or (x <= wall_width and
        #                                                            not self.at_door(unit, x, y))


    def neighbors(self, point, map_width, map_height, objs):
        """Generate walkable neighboring points avoiding collisions with objects."""
        step = 0.150
        directions = [(0, step), (step, 0), (0, -step), (-step, 0)]  # Adjacent squares: N, E, S, W
        x, y = point
        
        results = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
           # if 0 <= nx < map_width and 0 <= ny < map_height and all(self.collision(nx, ny, self.size[0], self.size[1], obj[]) for obj in objs):
            if 0 <= nx < map_width and 0 <= ny < map_height and all(self.objects_overlap(nx, ny, self.size[0], self.size[1], obj['position'][0],
                                                                                           obj['position'][1], obj['width'], obj['height']) for obj in objs) and  self.hits_wall( nx, ny):
                results.append((nx, ny))
        #print(results)
        return results
    def is_close_enough(self, current, goal, tolerance=0.15, is_item = True):
        """Check if the current position is within tolerance of the goal position."""
        if is_item is not None:
            tolerance = 0.6
            return (abs(current[0] - goal[0]) < tolerance - 0.15  and abs(current[1] - goal[1]) < tolerance +0.05 )

        else:
            return (abs(current[0] - goal[0]) < tolerance and abs(current[1] - goal[1]) < tolerance)
    def distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar(self, start, goal, objs, map_width, map_height, is_item = True):
        """Perform the A* algorithm to find the shortest path from start to goal."""
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        distance = 1000
   
        while not frontier.empty():

            current = frontier.get()
            #print(current, goal)
            # if distance > self.distance(current, goal):
            #     distance = self.distance(current, goal)
            #     print("getting closer: ", distance)
            if self.is_close_enough(current, goal, is_item=is_item):
                break

            for next in self.neighbors(current, map_width, map_height, objs):
                new_cost = cost_so_far[current] + 0.15  # Assume cost between neighbors is 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current

        # Reconstruct path
        if self.is_close_enough(current, goal, is_item=is_item):
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        return None  # No path found
    # PlayerAction.NORTH: (Direction.NORTH, (0, -1), 0),
    # PlayerAction.SOUTH: (Direction.SOUTH, (0, 1), 1),
    # PlayerAction.EAST: (Direction.EAST, (1, 0), 2),
    # PlayerAction.WEST: (Direction.WEST, (-1, 0), 3)
    def from_path_to_actions(self, path):
        """Convert a path to a list of actions."""
        # if the current direction is not the same as the direction of the first step in the path, add a TURN action
        # directions = [(0, step), (step, 0), (0, -step), (-step, 0)]  # Adjacent squares: N, E, S, W
        actions = []
        cur_dir = self.current_direction

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            if x2 > x1:
                if cur_dir != '2':
                    actions.append('EAST')
                    cur_dir = '2'
                    actions.append('EAST' )
                else:
                    actions.append('EAST')
            elif x2 < x1:
                if cur_dir != '3':
                    actions.append('WEST')
                    cur_dir = '3'
                    actions.append('WEST')
                else:
                    actions.append('WEST')
            elif y2 < y1:
                if cur_dir != '0':
                    actions.append('NORTH')
                    cur_dir = '0'
                    actions.append('NORTH')
                else:
                    actions.append('NORTH')
            elif y2 > y1:
                if cur_dir != '1':
                    actions.append('SOUTH')
                    cur_dir = '1'
                    actions.append('SOUTH')
                else:
                    actions.append('SOUTH')
        return actions

    def face_item(self, goal_x, goal_y):
        x, y = self.player['position']
        cur_dir = self.current_direction
        #print("info: ", cur_dir, y, goal_y)
        if goal_y < y:
            if cur_dir != '0':
                self.step('NORTH')
                dis = abs(goal_y - y)
                while dis> 0.75:
                    self.step('NORTH')
                    if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                        break
                    else:
                        dis = abs(goal_y - self.player['position'][1])
                return 'NORTH'
        elif goal_y > y:
            if cur_dir != '1':
                self.step('SOUTH')
                dis = abs(goal_y - y)
                while dis > 0.75:
                    self.step('SOUTH')
                    if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                        break
                    else:
                        dis = abs(goal_y - self.player['position'][1])
                return 'SOUTH'

    def perform_actions(self, actions):
        """Perform a list of actions."""
        for action in actions:
            self.step(action)
        return self.obs
    def change_direction(self, direction):
        cur_dir = self.current_direction
        if direction == 'NORTH':
            if cur_dir != '0':
                self.step('NORTH')
                return 'NORTH'
        elif direction == 'SOUTH':
            if cur_dir != '1':
                self.step('SOUTH')
                return 'SOUTH'
        elif direction == 'EAST':
            if cur_dir != '2':
                self.step('EAST')
                return 'EAST'
        elif direction == 'WEST':
            if cur_dir != '3':
                self.step('WEST')
                return 'WEST'
def find_item_position(data, item_name):
    """
    Finds the position of an item based on its name within the shelves section of the data structure.

    Parameters:
        data (dict): The complete data structure containing various game elements including shelves.
        item_name (str): The name of the item to find.

    Returns:
        list or None: The position of the item as [x, y] or None if the item is not found.
    """
    # Loop through each shelf in the data
    for shelf in data['observation']['shelves']:
        if shelf['food_name'] == item_name:
            return shelf['position']
    return None

# {'command_result': {'command': 'RESET', 'result': 'SUCCESS', 'message': '', 'stepCost': 0}, 'observation': {'players': [{'index': 0, 'position': [1.2, 15.6], 'width': 0.6, 'height': 0.4, 'sprite_path': None, 'direction': 2, 'curr_cart': -1, 'shopping_list': ['chocolate milk'], 'list_quant': [1], 'holding_food': None, 'bought_holding_food': False, 'budget': 100, 'bagged_items': [], 'bagged_quant': []}], 'carts': [], 'baskets': [], 'registers': [{'height': 2.5, 'width': 2.25, 'position': [1, 4.5], 'num_items': 0, 'foods': [], 'food_quantities': [], 'food_images': [], 'capacity': 12, 'image': 'images/Registers/registersA.png', 'curr_player': None}, {'height': 2.5, 'width': 2.25, 'position': [1, 9.5], 'num_items': 0, 'foods': [], 'food_quantities': [], 'food_images': [], 'capacity': 12, 'image': 'images/Registers/registersB.png', 'curr_player': None}], 'shelves': [{'height': 1, 'width': 2, 'position': [5.5, 1.5], 'food': 'milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'milk'}, {'height': 1, 'width': 2, 'position': [7.5, 1.5], 'food': 'milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'milk'}, {'height': 1, 'width': 2, 'position': [9.5, 1.5], 'food': 'chocolate milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_chocolate.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'chocolate milk'}, {'height': 1, 'width': 2, 'position': [11.5, 1.5], 'food': 'chocolate milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_chocolate.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'chocolate milk'}, {'height': 1, 'width': 2, 'position': [13.5, 1.5], 'food': 'strawberry milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_strawberry.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'strawberry milk'}, {'height': 1, 'width': 2, 'position': [5.5, 5.5], 'food': 'apples', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/apples.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'apples'}, {'height': 1, 'width': 2, 'position': [7.5, 5.5], 'food': 'oranges', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/oranges.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'oranges'}, {'height': 1, 'width': 2, 'position': [9.5, 5.5], 'food': 'banana', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/banana.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'banana'}, {'height': 1, 'width': 2, 'position': [11.5, 5.5], 'food': 'strawberry', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/strawberry.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'strawberry'}, {'height': 1, 'width': 2, 'position': [13.5, 5.5], 'food': 'raspberry', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/raspberry.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'raspberry'}, {'height': 1, 'width': 2, 'position': [5.5, 9.5], 'food': 'sausage', 'price': 4, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/sausage.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'sausage'}, {'height': 1, 'width': 2, 'position': [7.5, 9.5], 'food': 'steak', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_01.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'steak'}, {'height': 1, 'width': 2, 'position': [9.5, 9.5], 'food': 'steak', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_02.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'steak'}, {'height': 1, 'width': 2, 'position': [11.5, 9.5], 'food': 'chicken', 'price': 6, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'chicken'}, {'height': 1, 'width': 2, 'position': [13.5, 9.5], 'food': 'ham', 'price': 6, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/ham.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'ham'}, {'height': 1, 'width': 2, 'position': [5.5, 13.5], 'food': 'brie cheese', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_01.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'brie cheese'}, {'height': 1, 'width': 2, 'position': [7.5, 13.5], 'food': 'swiss cheese', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_02.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'swiss cheese'}, {'height': 1, 'width': 2, 'position': [9.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [11.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [13.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [5.5, 17.5], 'food': 'garlic', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/garlic.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'garlic'}, {'height': 1, 'width': 2, 'position': [7.5, 17.5], 'food': 'leek', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/leek_onion.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'leek'}, {'height': 1, 'width': 2, 'position': [9.5, 17.5], 'food': 'red bell pepper', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/bell_pepper_red.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'red bell pepper'}, {'height': 1, 'width': 2, 'position': [11.5, 17.5], 'food': 'carrot', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/carrot.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'carrot'}, {'height': 1, 'width': 2, 'position': [13.5, 17.5], 'food': 'lettuce', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/lettuce.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'lettuce'}, {'height': 1, 'width': 2, 'position': [5.5, 21.5], 'food': 'avocado', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/avocado.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'avocado'}, {'height': 1, 'width': 2, 'position': [7.5, 21.5], 'food': 'broccoli', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/broccoli.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'broccoli'}, {'height': 1, 'width': 2, 'position': [9.5, 21.5], 'food': 'cucumber', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cucumber.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cucumber'}, {'height': 1, 'width': 2, 'position': [11.5, 21.5], 'food': 'yellow bell pepper', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/bell_pepper_yellow.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'yellow bell pepper'}, {'height': 1, 'width': 2, 'position': [13.5, 21.5], 'food': 'onion', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/onion.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'onion'}], 'cartReturns': [{'height': 6, 'width': 0.7, 'position': [1, 18.5], 'quantity': 6}, {'height': 6, 'width': 0.7, 'position': [2, 18.5], 'quantity': 6}], 'basketReturns': [{'height': 0.2, 'width': 0.3, 'position': [3.5, 18.5], 'quantity': 12}], 'counters': [{'height': 2.25, 'width': 1.5, 'position': [18.25, 4.75], 'food': 'prepared foods', 'price': 15}, {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 'food': 'fresh fish', 'price': 12}, {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 'food': 'fresh fish', 'price': 12}]}, 'step': 1, 'gameOver': False, 'violations': ''}
# [11.5, 17.5]
# hang@jstaley-XPS-8940:~/TA/propershopper-1$ python astar_path_planner.py 
# action_commands:  ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']
# ['sausage', 'milk', 'prepared foods', 'onion', 'fresh fish', 'apples', 'oranges', 'steak']
# {'command_result': {'command': 'RESET', 'result': 'SUCCESS', 'message': '', 'stepCost': 0}, 'observation': {'players': [{'index': 0, 'position': [1.2, 15.6], 'width': 0.6, 'height': 0.4, 'sprite_path': None, 'direction': 2, 'curr_cart': -1, 'shopping_list': ['sausage', 'milk', 'prepared foods', 'onion', 'fresh fish', 'apples', 'oranges', 'steak'], 'list_quant': [2, 3, 1, 1, 1, 1, 1, 1], 'holding_food': None, 'bought_holding_food': False, 'budget': 100, 'bagged_items': [], 'bagged_quant': []}], 'carts': [], 'baskets': [], 'registers': [{'height': 2.5, 'width': 2.25, 'position': [1, 4.5], 'num_items': 0, 'foods': [], 'food_quantities': [], 'food_images': [], 'capacity': 12, 'image': 'images/Registers/registersA.png', 'curr_player': None}, {'height': 2.5, 'width': 2.25, 'position': [1, 9.5], 'num_items': 0, 'foods': [], 'food_quantities': [], 'food_images': [], 'capacity': 12, 'image': 'images/Registers/registersB.png', 'curr_player': None}], 'shelves': [{'height': 1, 'width': 2, 'position': [5.5, 1.5], 'food': 'milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'milk'}, {'height': 1, 'width': 2, 'position': [7.5, 1.5], 'food': 'milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'milk'}, {'height': 1, 'width': 2, 'position': [9.5, 1.5], 'food': 'chocolate milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_chocolate.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'chocolate milk'}, {'height': 1, 'width': 2, 'position': [11.5, 1.5], 'food': 'chocolate milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_chocolate.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'chocolate milk'}, {'height': 1, 'width': 2, 'position': [13.5, 1.5], 'food': 'strawberry milk', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/milk_strawberry.png', 'shelf_image': 'images/Shelves/fridge.png', 'food_name': 'strawberry milk'}, {'height': 1, 'width': 2, 'position': [5.5, 5.5], 'food': 'apples', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/apples.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'apples'}, {'height': 1, 'width': 2, 'position': [7.5, 5.5], 'food': 'oranges', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/oranges.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'oranges'}, {'height': 1, 'width': 2, 'position': [9.5, 5.5], 'food': 'banana', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/banana.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'banana'}, {'height': 1, 'width': 2, 'position': [11.5, 5.5], 'food': 'strawberry', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/strawberry.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'strawberry'}, {'height': 1, 'width': 2, 'position': [13.5, 5.5], 'food': 'raspberry', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/raspberry.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'raspberry'}, {'height': 1, 'width': 2, 'position': [5.5, 9.5], 'food': 'sausage', 'price': 4, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/sausage.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'sausage'}, {'height': 1, 'width': 2, 'position': [7.5, 9.5], 'food': 'steak', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_01.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'steak'}, {'height': 1, 'width': 2, 'position': [9.5, 9.5], 'food': 'steak', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_02.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'steak'}, {'height': 1, 'width': 2, 'position': [11.5, 9.5], 'food': 'chicken', 'price': 6, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/meat_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'chicken'}, {'height': 1, 'width': 2, 'position': [13.5, 9.5], 'food': 'ham', 'price': 6, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/ham.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'ham'}, {'height': 1, 'width': 2, 'position': [5.5, 13.5], 'food': 'brie cheese', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_01.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'brie cheese'}, {'height': 1, 'width': 2, 'position': [7.5, 13.5], 'food': 'swiss cheese', 'price': 5, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_02.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'swiss cheese'}, {'height': 1, 'width': 2, 'position': [9.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [11.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [13.5, 13.5], 'food': 'cheese wheel', 'price': 15, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cheese_03.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cheese wheel'}, {'height': 1, 'width': 2, 'position': [5.5, 17.5], 'food': 'garlic', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/garlic.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'garlic'}, {'height': 1, 'width': 2, 'position': [7.5, 17.5], 'food': 'leek', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/leek_onion.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'leek'}, {'height': 1, 'width': 2, 'position': [9.5, 17.5], 'food': 'red bell pepper', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/bell_pepper_red.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'red bell pepper'}, {'height': 1, 'width': 2, 'position': [11.5, 17.5], 'food': 'carrot', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/carrot.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'carrot'}, {'height': 1, 'width': 2, 'position': [13.5, 17.5], 'food': 'lettuce', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/lettuce.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'lettuce'}, {'height': 1, 'width': 2, 'position': [5.5, 21.5], 'food': 'avocado', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/avocado.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'avocado'}, {'height': 1, 'width': 2, 'position': [7.5, 21.5], 'food': 'broccoli', 'price': 1, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/broccoli.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'broccoli'}, {'height': 1, 'width': 2, 'position': [9.5, 21.5], 'food': 'cucumber', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/cucumber.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'cucumber'}, {'height': 1, 'width': 2, 'position': [11.5, 21.5], 'food': 'yellow bell pepper', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/bell_pepper_yellow.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'yellow bell pepper'}, {'height': 1, 'width': 2, 'position': [13.5, 21.5], 'food': 'onion', 'price': 2, 'capacity': 12, 'quantity': 12, 'food_image': 'images/food/onion.png', 'shelf_image': 'images/Shelves/shelf.png', 'food_name': 'onion'}], 'cartReturns': [{'height': 6, 'width': 0.7, 'position': [1, 18.5], 'quantity': 6}, {'height': 6, 'width': 0.7, 'position': [2, 18.5], 'quantity': 6}], 'basketReturns': [{'height': 0.2, 'width': 0.3, 'position': [3.5, 18.5], 'quantity': 12}], 'counters': [{'height': 2.25, 'width': 1.5, 'position': [18.25, 4.75], 'food': 'prepared foods', 'price': 15}, {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 'food': 'fresh fish', 'price': 12}, {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 'food': 'fresh fish', 'price': 12}]}, 'step': 1, 'gameOver': False, 'violations': ''}
# [11.5, 17.5]



action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

print("action_commands: ", action_commands)

     # Connect to Supermarket
HOST = '127.0.0.1'
PORT = 9000
sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_game.connect((HOST, PORT))
sock_game.send(str.encode("0 RESET"))  # reset the game
state = recv_socket_data(sock_game)
game_state = json.loads(state)
shopping_list = game_state['observation']['players'][0]['shopping_list']
shopping_quant = game_state['observation']['players'][0]['list_quant']
player = Agent(socket_game=sock_game, env=game_state)
cartReturns = [2, 18.5]
basketReturns = [3.5, 18.5]
registerReturns_1 = [2, 4.5]
registerReturns_2 = [2, 9.5]
print(shopping_list)

offset = 1

# shopping_list = ['fresh fish', 'prepared foods']


if len(shopping_list) > 0:
    print("go for basket")
    player.perform_actions(player.from_path_to_actions(player.astar((player.player['position'][0], player.player['position'][1]),
    (basketReturns[0], basketReturns[1] ), objs, 20, 25)))
    player.face_item(basketReturns[0], basketReturns[1])
    player.step('INTERACT')
    player.step('INTERACT')
    for item in shopping_list:
        y_offset = 0
        print("go for item: ", item)
        item_pos = find_item_position(game_state, item)
        if item != 'prepared foods' and item != 'fresh fish': 
            item_pos = find_item_position(game_state, item)
            #item = update_position_to_center(item_pos)
        else:
            if item == 'prepared foods':
                item_pos = [18.25, 4.75]
            else:
                item_pos = [18.25, 10.75]
        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 3
        # print("item_pos: ", item_pos)
        path  = player.astar((player.player['position'][0], player.player['position'][1]),
                              (item_pos[0] + offset, item_pos[1] + y_offset), objs, 20, 25)
        if path == None:
            continue
        player.perform_actions(player.from_path_to_actions(path))
        player.face_item(item_pos[0] + offset, item_pos[1])
        #player.face_item(item_pos[0] + offset, item_pos[1])
        for i in range(shopping_quant[shopping_list.index(item)]):
            player.step('INTERACT')
            if item == 'prepared foods' and item == 'fresh fish':
                player.step('INTERACT')

        #print(player.obs)
    #print(player.obs['players'][0]['shopping_list'])

    # go to closer register
    if player.player['position'][1] < 7:
        path = player.astar((player.player['position'][0], player.player['position'][1]),
                            (registerReturns_1[0] + offset, registerReturns_1[1]), objs, 20, 25)
        if path == None:
            print("no path to register")
        player.perform_actions(player.from_path_to_actions(path))
        #player.face_item(registerReturns_1[0] + offset, registerReturns_1[1])
    else:
        path = player.astar((player.player['position'][0], player.player['position'][1]),
                            (registerReturns_2[0] + offset, registerReturns_2[1]), objs, 20, 25)
        if path == None:
            print("no path to register")
        player.perform_actions(player.from_path_to_actions(path))
        #player.face_item(registerReturns_2[0] + offset, registerReturns_2[1])
  
    player.step('INTERACT')
    player.step('INTERACT')
    player.step('INTERACT')
    print(player.obs)
    

