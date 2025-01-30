from enums.direction import Direction
from copy import deepcopy

def obj_collision(obj,  x_position, y_position, x_margin=0.55, y_margin=0.55):
    return obj.position[0] - x_margin < x_position < obj.position[0] + obj.width + x_margin and \
           obj.position[1] - y_margin < y_position < obj.position[1] + obj.height + y_margin


def overlap(x1, y1, width_1, height_1, x2, y2, width_2, height_2):
    return not (x1 > x2 + width_2 or x2 > x1 + width_1 or y1 > y2 + height_2 or y2 > y1 + height_1)


def objects_overlap(obj1, obj2):
    return overlap(obj1.position[0], obj1.position[1], obj1.width, obj1.height,
                   obj2.position[0], obj2.position[1], obj2.width, obj2.height)


def pos_collision(x1, y1,  x2, y2, x_margin, y_margin):
    return x1 - x_margin < x2 < x1 + x_margin and y1 - y_margin < y2 < y1 + y_margin

def project_collision(obj:dict|list|tuple, state, direction: Direction, dist=0.4, buffer=0.0):
    """Project collision. This should only be used when the player is likely far from the target item they want to interact with. Otherwise, the player might get stuck turning back and forth in a corner formed by static obstacles 

    Args:
        obj (dict): most likely the player
        state (dict): game state
        direction (Direction): directional command
        dist (float, optional): distance the obj is about to travel. Defaults to 0.4.
        buffer (float, optional): buffer between objects in the env. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    obj_copy = deepcopy(obj)

    if direction == Direction.NORTH:
        obj_copy['position'][1] -= dist
        if obj_copy['position'][1] < 2.1:
            return True
    elif direction == Direction.EAST:
        obj_copy['position'][0] += dist
        if obj_copy['position'][0] > 18.5:
            return True
    elif direction == Direction.SOUTH:
        obj_copy['position'][1] += dist
        if obj_copy['position'][1] > 24:
            return True
    elif direction == Direction.WEST:
        obj_copy['position'][0] -= dist
        if obj_copy['position'][0] < 0.55:
            # if abs(obj_copy['position'][1] - exit_pos[1]) < STEP: # at the exit
            #     return False
            return True

    for key, value in state['observation'].items():
        for i, item in enumerate(value):
            if key == 'players':#for players, pretend that they are wider and taller than they actually are to stay away
                if (overlap(obj_copy['position'][0], obj_copy['position'][1], obj_copy['width'], obj_copy['height'],
                            item['position'][0] - 0.7, item['position'][1] - 0.7, item['width'] + 0.7 + 0.7, item['height'] + 0.7 + 0.7)):
                    if not (obj_copy == item or (
                        'index' in item.keys() and 'index' in obj_copy.keys() and item['index'] == obj_copy['index'])):
                    
                        print("projected collision with: ", {key})
                        return True
            else:
                if key == 'carts':
                    if 'curr_cart' in obj_copy and i == obj_copy['curr_cart']:
                        continue # don't project collision with the cart we are currently holding
                if key == 'baskets':
                    if 'curr_basket' in obj_copy and i == obj_copy['curr_basket']:
                        continue # don't project collision with the basket we are currently holding
                if (overlap(obj_copy['position'][0], obj_copy['position'][1], obj_copy['width'], obj_copy['height'],
                            item['position'][0], item['position'][1], item['width'] + buffer, item['height'] + buffer)):
                    if not (obj_copy == item or (
                        'index' in item.keys() and 'index' in obj_copy.keys() and item['index'] == obj_copy['index'])):
                    
                        print("projected collision with: ", {key})
                        return True
    return False

def can_interact_default(obj, player, range=0.25):
    if player.direction == Direction.NORTH:
        return obj.collision(player, player.position[0], player.position[1] - range)
    elif player.direction == Direction.SOUTH:
        return obj.collision(player, player.position[0], player.position[1] + range)
    elif player.direction == Direction.WEST:
        return obj.collision(player, player.position[0] - range, player.position[1])
    elif player.direction == Direction.EAST:
        return obj.collision(player, player.position[0] + range, player.position[1])
    return False