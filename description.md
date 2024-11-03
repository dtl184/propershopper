# Start Training

Start environment `$python socket_env.py --stay_alive --headless --save_video $(save_video) --record_path $(record_path) `.

Run `$python socket_train_whole.py` to start training.


# File Structure

`planner.py` defines a hyper planner responsible for decomposing whole task and selecting agents for each sub-task.

`constants.py` stores environment information, e.g. object positions, obstacle regions. 

`planning_agent.py` defines navigation agents. This is implemented with planning conditioned on target object. 

`get_obj_agent.py` defines agents specific for basket tasks. They are trained with Q-learning to put a designated object into basket. 

`cart_get_obj_agent.py` defines agents specific for cart tasks. They are trained with Q-learning to put a designated object into cart. 

`checkout_agent.py` defines agents specific for basket tasks. They are trained with Q-learning to do checkout.

`cart_checkout_agent.py` defines agents specific for cart tasks. They are trained with Q-learning to do checkout. 

`socket_train_whole.py` initializes a task, does task decomposition and trains relevant agents in the task. 

# Hyper Planner 

Hyper planner is responsible for parsing the shopping list and generating a whole task. It first counts the number of items to buy and decides whether to use a cart (>6) or a basket. Then it generates a plan: navigate to basket/cart, get a basket/cart, navigate to the item, get the item for N (number of this item needed) times, navigate to checkout, do checkout, navigate to exit. It also initializes all the agents and select one agent for each step. It ensures `OneCartOnlyNorm()`, `OneBasketOnlyNorm()`, `WrongShelfNorm()`, `EntranceOnlyNorm()`, `ReturnBasketNorm()`, `BasketItemQuantNorm(basket_max=<max items>)`, `CartItemQuantNorm(cart_min=<min items>)`, `UnattendedCheckoutNorm()`, `TookTooManyNorm()`.

# Navigationg Agents

Navigation is implemented with planning instead of reinforcement learning for simplicity. Each navigation agent is conditioned on one goal object. It discritizes the environment space into grids and uses BFS to calculate distances to the target location. They are responsible for navigating to somewhere near the objects to simplify the RL training process of next steps. The bounding boxes of relevant obstacles are enlarged to ensure `ObjectCollisionNorm()`, `WallCollisionNorm()`.

# Get-Object Agents

Get-object agents are responsible for getting one target object into cart of basket. In general they are able to navigate to the object from anywhere then interact with it. In practice they are always initialized after navigating close enough to the target. Each agent also maintains a separate norm Q-table to ensure `ObjectCollisionNorm()`, `WallCollisionNorm()`.

## Basket Get-Object Agents

GetObjAgent defined in `get_obj_agent.py` is specific for basket tasks. The state space is defined by discritized position. While using a basket, to get an object is to interact with it. The action space is defined by four directions and 'INTERACT'. This agent only does this interaction once. The number of items to get is given by the iteration of this action in the plan to satisfy `TookTooManyNorm()`. The norm Q-table is able to store other potential norms. 

A coarse heuristic reward calculated with Euclidean distance is given each time the agent moves toward the object. Since the initialization is finished by navigation, the agent is near the object and a coarse reward would be sufficient. After getting the object successfully, a high positive reward is given, suggesting the end of the sub-task. Negative rewards are given once any norm is violated.

## Cart Get-Object Agents

CartGetObjAgent defined in `cart_get_obj_agent.py` is specific for cart tasks. The state space is defined by discritized position and whether the agent is holding the target object. The agent has to hold one  This agent only does this interaction once. The number of items to get is given by the iteration of this action in the plan to satisfy `TookTooManyNorm()`. The norm Q-table is able to store other potential norms. 

The reward function for cart agents are two-fold. A coarse heuristic reward calculated with Euclidean distance is given each time the agent moves toward the object before getting it. A high intermediate reward is given once the agent gets the object. Then a coarse heuristic reward calculated with Euclidean distance between agent and the cart is given. Since the initialization is finished by navigation, the agent is near the object and a coarse reward would be sufficient. After toggling the cart again, a high positive reward is given, suggesting the end of the sub-task. Negative rewards are given once any norm is violated. 

# Checkout Agents

Checkout agents are just like Get-object agents. They are trained with Q-learning in a model-free way. They are initialized after navigation to ensure simplicity. And they maintain norm Q-tables to learn dynamically to satisfy norms.

## Basket Checkout Agents

Checkout agents defined in `checkout_agent.py` is specific for basket tasks. The state space is discritized position. The action space is four directions and 'INTERACT' since no other skills are needed for baskets.

Unlike other objects in the environmemt, the position of checkout center is fixed and are known to be closed to the planned position. Therefore a sparse reward is sufficient. A high reward is given once the agents finishes interaction (checked with `purchased_contents`). Negative rewards are given once any norm is violated. 

# Cart Checkout Agents 

Cart checkout agents defined in `cart_checkout_agent.py` is specific for cart tasks. The state space is discritized position, the list of purchased items, the list of numbers of purchased items, the list of contents in the cart. The action space is four directions, 'TOGGLE_CART', 'SELECT', 'INTERACT'.

Unlike other objects in the environmemt, the position of checkout center is fixed and are known to be closed to the planned position. A high intermediate is given once the agent takes out unpayed items, pays for the items. A high reward is given once the agents finishes all the interaction (checked with `purchased_contents`). Negative rewards are given once any norm is violated. 