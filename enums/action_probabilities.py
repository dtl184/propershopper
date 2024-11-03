# Note: This really isn't an enum but a constant. Should this be placed in a different folder?
from enums.player_action import PlayerAction

Action_Probabilities = {
    PlayerAction.NOP : {PlayerAction.NOP: 1},
    PlayerAction.NORTH : {PlayerAction.NORTH: 1, PlayerAction.SOUTH: 0, PlayerAction.EAST: 0, PlayerAction.WEST: 0, PlayerAction.NOP: 0},
    PlayerAction.SOUTH : {PlayerAction.NORTH: 0, PlayerAction.SOUTH: 1, PlayerAction.EAST: 0, PlayerAction.WEST: 0, PlayerAction.NOP: 0},
    PlayerAction.EAST : {PlayerAction.NORTH: 0, PlayerAction.SOUTH: 0, PlayerAction.EAST: 1, PlayerAction.WEST: 0, PlayerAction.NOP: 0},
    PlayerAction.WEST : {PlayerAction.NORTH: 0, PlayerAction.SOUTH: 0, PlayerAction.EAST: 0, PlayerAction.WEST: 1, PlayerAction.NOP: 0},
    PlayerAction.INTERACT : {PlayerAction.INTERACT: 1, PlayerAction.NOP: 0},
    PlayerAction.TOGGLE : {PlayerAction.TOGGLE: 1, PlayerAction.NOP: 0},
    PlayerAction.CANCEL : {PlayerAction.CANCEL: 1, PlayerAction.NOP: 0},
    PlayerAction.PICKUP : {PlayerAction.PICKUP: 1, PlayerAction.NOP: 0},
    PlayerAction.RESET : {PlayerAction.RESET: 1},
}