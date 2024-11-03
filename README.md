# PROPER SHOPPER

Original developers: [Teah Markstone](https://github.com/teahmarkstone/) and [Daniel Kasenberg](https://github.com/dkasenberg/)

Additional developers: [Matthias Scheutz](https://github.com/mscheutz/)

## Task setting

An autonomous agent has to perform a shopping task in a simulated supermarket environment, gathering food items on a shopping list in a cart or basket, paying for them at the cash register and leaving the store with the purchased items.  The simulation is based on [Gymnasium].(https://gymnasium.farama.org/) and can be run manually with keyboard input, or with autonomous agents (in Python and Java) connected via a socket connection.

## Dependencies:
* python 3.12+
* pip (to install any required libraries, e.g., "pip install pygame" or "pip install gymnasium")
* pygame 2.5.0+
* gymnasium 0.29.1

## Installing
Clone the repo

```
git clone https://github.com/mscheutz/propershopper.git
```

## Running keyboard input
Running the simulation with keyboard input is a good way to get a sense of the physics of the simulation and how interaction with objects works.

To run keyboard input, while in the "proppershopper" directory, run

```
<python-command> socket_env.py --keyboard_input
```

where <python-command> is your command for running python.

## Running with an autonomous agent through a socket

From a terminal in the proper shopper directory, use your python command to run socket_env.py (without any arguments):

```
<python-command> socket_env.py
```
For debugging you can start "socket_env.py" with a --file <path_to_file> flag that will load in the state from the file at the specified path.  You can use your keyboard input (with "<yourpythoncommand> socket_env.py --keyboard_input") to press the 's' key while running the simulation.  If you then enter a filename in the terminal from which they're running the simulation (e.g., "myseed.txt"), then the current state of the game will be saved to "myseed.txt" and can later be reloaded using --file flag.

### Running the simulation with Python agents
Take a look at the "socket_agent.py" file which has an example of how to interact with the simulation through a socket.  The format for sending actions is "<agentnumber> <actioncommand>" (the commands are described below).  Here is the snippet:

```
# build an action packet assuming this is the only agent in the game
action = "0 " + random.choice(action_commands)

# send it to the environment
sock_game.send(str.encode(action))

# get an observation back
output = recv_socket_data(sock_game)  # get observation from env
output = json.loads(output)
```
You can then run it (assuming the simulation is already running in another terminal) using
```
python socket_agent.py
```

### Running the simulation with Java agents
Put all of your agent code into a class called Agent.java (a template is in the repo) which has the following structure:

```
import com.supermarket.*;

public class Agent extends SupermarketComponentImpl {
	public Agent() {
		super();
		shouldRunExecutionLoop = true;
	}

	@Override
	protected void executionLoop() {
		// this is called every 100ms
		// put your code in here, e.g.
	      goSouth();
	}
}
```

Assuming you put the class in the same directory as you Agent.jar, then you can compile it in that directory using:

```
	javac -cp Agent.jar Agent.java
```

and you can run using the following (after you first made sure that the socket is running, won’t work without it!)

```
	java -cp .:Agent.jar Agent Agent <any other args>
```

(Use “;” instead of “:” for Windows!)


## Command Line Options
When running through the socket or from keyboard input, there are a number of command line arguments to specify aspects of the supermarket.

*  ``--num_players=<number of players>`` specifies the number of players in the environment
*  ``--port=<port number>`` specifies which port to bind to
*  ``--headless`` sets the environement to  headless mode
*  ``--keyboard_input`` sets the environment to use keyboard input
*  ``--file=<filename>`` loads the state from the specified file
*  ``--follow=<player ID>`` specifies which player to follow
*  ``--random_start`` sets players to start in random locations
*  ``--render_number`` sets environment to render player ID above players
*  ``--render_messages`` sets environment to render messages when interacting with objects
*  ``--bagging`` sets checkout option to use bagging
*  ``--player_sprites=<filename>`` sets player sprites to specified sprites
*  ``--record_path=<path>`` sets path for files when record action is used

## Norm Monitoring
When run through keyboard input or through the socket, a norm monitor gym wrapper is automatically used and monitors all implemented norms. Currently we have 26 norm monitors implemented:
*  ``BasketTheftNorm()`` Logs norm violation if player steals another player's basket
*  ``WrongShelfNorm()`` Logs norm violation if player puts an item back on the wrong shelf
*  ``ShopliftingNorm()`` Logs norm violation if player leaves the store with unbought items in their inventory
*  ``PlayerCollisionNorm()`` Logs norm violation if player collides with another player
*  ``ObjectCollisionNorm()`` Logs norm violation if player collides with an object
*  ``WallCollisionNorm()`` Logs norm violation if player collides with a wall
*  ``BlockingExitNorm()`` Logs norm violation if player is standing in front of an exit
*  ``EntranceOnlyNorm()`` Logs norm violation if player exits store through the entrance
*  ``UnattendedCartNorm()`` Logs norm violation if player leaves a cart at a distance of 5 or more for 5 or more timesteps
*  ``UnattendedBasketNorm()`` Logs norm violation if player leaves a cart at a distance of 5 or more for 5 or more timesteps
*  ``OneCartOnlyNorm()`` Logs norm violation if player takes more than one cart from the cart return
*  ``OneBasketOnlyNorm()`` Logs norm violation if player takes more than one basket from the basket return
*  ``PersonalSpaceNorm(dist_threshold=<distance>)`` Logs norm violation if player gets too close to another player with specified distance threshold
*  ``InteractionCancellationNorm()`` Logs norm violation if player cancels interaction
*  ``LeftWithBasketNorm()`` Logs norm violation if bagging option is being used and the player leaves the store with a basket
*  ``ReturnBasketNorm()`` Logs norm violation if player does not return all of their baskets before leaving the store
*  ``ReturnCartNorm()`` Logs norm violation if player does not return all of their carts before leaving the store
*  ``WaitForCheckoutNorm()`` Logs norm violation if player cuts in line at checkout
*  ``ItemTheftFromCartNorm()`` Logs norm violation if player takes item out of another player's cart
*  ``ItemTheftFromBasketNorm()`` Logs norm violation if player takes item out of another player's basket
*  ``AdhereToListNorm()`` Logs norm violation if player picks up an item that is not on their shopping list
*  ``TookTooManyNorm()`` Logs norm violation if player takes more of one kind of food item than is on their list
*  ``BasketItemQuantNorm(basket_max=<max items>)`` Logs norm violation if player takes a basket when the amount of items on their shopping list exceeds the set max capacity for the basket
*  ``CartItemQuantNorm(cart_min=<min items>)`` Logs norm violation if player takes a cart when the amount of items on their shopping list is below the minimum number of items that require a cart
*  ``UnattendedCheckoutNorm()`` Logs norm violation if player leaves a cart in  the checkout zone for too long (non-bagging) or walks away from checkout counter while their items are on the counter (bagging)

To use the norm wrapper gym environment, pass in a supermarket environment and a list of the above norms to be monitored. The violations are returned as an observation from the norm environment. In keyboard input, the norms are printed out to the terminal.

## Playing the game

### Actions and environment dynamics
Below is a table of the gym ids of the actions available in this environment (left column), the corresponding action (middle column), and the key to press in keyboard input mode to perform the action (right column):

| gym id | Action             | Key          |
|--------|--------------------|--------------|
| 0      | No-op                | N/A          |
| 1      | North              | Up arryow    |
| 2      | South              | Down arrow   |
| 3      | East               | Right arrow  |
| 4      | West               | Left arrow   |
| 5      | Interact           | Enter/Return |
| 6      | Toggle cart        | 'c'          |
| 7      | Cancel interaction | 'b'          |
| 8      | Pick Up            | N/A          |
| N/A    | View inventory     | 'i'          |
| N/A    | View shopping list | 'l'          |

Actions are formatted as tuples. In the pick-up action an agent can specify an index that corresponds with a food item.


While the best way to get a sense for how the game works is to try running it in keyboard input mode, here is a reasonably-detailed description of the actions available and what they do in different contexts:

* North, south, east, west: move the player around on the screen.
* No-op: does nothing. You probably won’t need to get the agent to perform this action.
* **Interact** with object: only works when the agent is adjacent to and facing a particular object in the game. Pauses the player’s ability to navigate while a message is displayed on screen; you must perform the “interact with action” or “cancel interaction” effect again to clear this message and continue running the game. When holding a shopping cart, the player cannot interact with any objects aside from the cart itself. Otherwise, effect depends on the object in question: 
  * Carts and Baskets:
    * The user may do this from in front of, behind, or beside a shopping cart/basket.
    * If the player is holding food, this action will put it in the cart/basket.
    * Otherwise, in keyboard input a menu of the items in the cart/basket will be displayed and the player can navigate it with up and down arrows. Use return key to take selected food item out of cart.
  * Cart return (bottom left corner of supermarket): if the player is not holding a shopping cart, this will cause the player to pick up a new cart. If they are holding a cart, it will be returned.
  * Basket return (next to cart return): if the player is not holding a basket, this will cause the player to pick up a new basket. If they are holding a basket, it will be returned.
  * Shelf:
    * If the player is currently holding food, this command will “put it back on the shelf” (so that the player is no longer holding it).
    * Otherwise, the player will pick up the shelf’s food item.
  * Food counters (right side of supermarket):
    * If the player is not currently holding food, this command will cause the agent to pick up the counter’s food (the top counter is “prepared foods”, the bottom counter is “fresh fish”).
    * This is a two-stage interaction, meaning that two consecutive messages are displayed (and thus the “interact with object” action must be performed three times total to complete the transaction and continue the game).
  * Checkout (left side of supermarket)
    * Non-bagging Option:     
      * Interacting with the checkout causes the purchase of (1) any unpurchased items that are currently in the player’s hand, and (2) any unpurchased items that are in carts that are directly above or below the checkout aisle.
      * Place your shopping cart either directly below or directly above the checkout if you want to purchase its contents.
      * This is a two-stage interaction, meaning that two consecutive messages are displayed (and thus the “interact with object” action must be performed three times total to complete the transaction and continue the game).
    * Bagging Option:
      * Interacting with the checkout renders a message saying to place items on counter. If player is holding a food item and interacts with the checkout counter, the food item will be placed on the counter.
      * If there are already food items on the counter and a player interacts, in keyboard input a menu will appear where a player can either select a food item to take off of the counter, exit the menu, or buy all of the items on the counter
      * In non-keyboard input, the interact action buys all the items on the counter. The agent can use the pick up action to pick up a specified food from the counter.
* **Pick up** item: used in gym environment to pick up food items from baskets, carts, and checkout counters. This action is only used through the environment, not with keyboard input. The second argument of the action specifies an index that corresponds to a food type in a list in the supermarket class. If the food item is in the interacted with object, the item will be removed and put into the player's hands.
* Cancel interaction: clears any messages caused by interacting with an object. For two-stage interactions (checkout and food counters) this will cancel the transaction in question (not obtain the counter’s food; not purchase cart contents). For one-stage interactions, this is equivalent to clearing the messages using “interact with object”.
* Toggle shopping cart: if the user is currently holding a shopping cart, this lets go of the cart. Otherwise, picks up the cart. Note that to pick up a cart, the user must be behind the shopping cart (adjacent to the handle).
* Display inventory:  displays a list of the food (a) that the player is holding, or (b) that is in the player’s cart, and the quantities of each food. Only available in keyboard mode; not necessary in gym/socket mode (since this information is already available in the observation).
* Display shopping list:  displays a list of the food on the player’s shopping list, and the quantities of each food. Only available in keyboard mode; not necessary in gym/socket mode (since this information is already available in the observation).

### General facts

* Objects are solid; you can’t walk through walls, shelves, shopping carts, or baskets
* When holding a shopping cart, that cart is an extension of you for collision purposes; the cart won’t pass through solid objects either
* Shopping lists are randomly generated when the game initializes, and have anywhere between 1 and 12 items
* Shopping carts can each carry a maximum of 12 food items
* Baskets can carry a maximum of 6 items
* There are 12 baskets and 12 carts in the environment. When all carts or baskets are taken from the cart/basket return, the object will still be solid but have no image so that players can still interact return their carts or baskets.
* Food items must (for now) be picked up and placed in the cart one at a time. If you need three apples, you must do the following three times: go to the shelf, pick up an apple, go to the cart, put the apple in the cart.
* To put food items in a basket, the player just needs to interact with a shelf while holding a basket.
* The player can leave their shopping cart or basket anywhere in the store, and can even have multiple shopping carts and baskets: any item in any of your carts/baskets counts towards your inventory
* You can only hold one shopping cart or basket at a time
* Putting food back on the wrong shelf does not mean you’ll pick up the wrong food when you interact with that shelf later (if you were in a grocery store in the canned foods aisle looking for soup and saw 15 cans of soup and one apple, would you pick up the apple?)
* In keyboard mode, the green number next to food items on a player's inventory is the number of purchased items of that food, and the red number is unpurchased.
* A red line will cross off an item on your shopping list when the correct amount of a certain food is in a player's inventory.
* You can’t leave a cart above or below one checkout, and then expect your food to be purchased when you interact with the other checkout. This should more or less go without saying.
* In the keyboard input mode, you can exit the game with the ESC key, or by leaving through the exit door. You can also Ctrl+C out of the Python program or “X out” of the window.
* In gym or socket mode, you can exit the game by leaving through the exit door, Ctrl+C in your terminal, or “X out” of the window. However, in socket mode, I recommend closing whatever process is interacting with the socket first (if you exit the simulation first, the socket might remain open and make it so that you can’t rerun the simulation for a minute or so).

### Observations
When you use it as a gym env or in socket mode, you're going to need to access the state observations.
* The observation keeps track of the states of the individual objects (players, carts, baskets, shelves, registers, cart returns, counters, checkouts).  There’s an array for each class of object.
  * All objects have a position (a 2D array x and y). Objects other than carts and players also have a width and height. Be careful in how you interpret this; just because a person is colliding with an object doesn’t mean that object.x < player.x < object.x + object.width, etc. Take a look at the collision() and canInteract() methods for various object classes to get a sense of how this actually works.
  * Shelves and counters have a particular food associated with them, represented as a string (“apples”, etc.)
  * Players, carts, and baskets are facing a particular direction (0=north, 1=south, 2=east, 3=west).
  * Players may be holding a particular food (field “holding_food”, will be null if not holding any food).
  * Players have a shopping list shopping_list of food strings (matching those on particular shelves). There’s a corresponding list list_quant of the number of each type of food on the list. For example, if player.shopping_list[i].equals(“apples”), then player.list_quant[i] is the number of apples on the shopping list.
The Player’s curr_cart is the index in observation.carts of the cart the player is holding, or -1 if the player is not holding a cart.
  * Each cart has an owner (who first picked the cart up from the cart return) and a separate variable last_held (who last touched the cart). This probably won’t matter in single-agent settings, but would matter in multi-agent games.
  * Each cart has a capacity. This’ll just be 12.
  * Each cart has variables contents and contents_quant (same format as a player’s shopping list, but for the unpurchased contents of a cart) and purchased_contents and purchased_quant (same format, but for the purchased contents of a cart).
 The above is true for baskets as well.
* Each player keeps track of their own interaction stage with an object:
  * If a player is not currently interacting with an object (i.e., if no interaction message is on the screen), interactive_stage=-1 and total_stages=0.
  * Otherwise, a player is interacting with some object; interactive_stage is the (zero-indexed) current stage, and total_stages will be 1 or 2 depending on if the interaction in question is a zero-stage interaction.
* The observation contains a little information about the “interactive stage” of the game, which can presumably help you remember to call interactWithObject() an appropriate number of times:
  * If the user is not currently interacting with an object (i.e., if no interaction message is on the screen), interactive_stage=-1 and total_stages=0.
  * Otherwise, the user is interacting with some object; interactive_stage is the (zero-indexed) current stage, and total_stages will be 1 or 2 depending on if the interaction in question is a zero-stage interaction.
* The observation has some helper methods that may come in handy. You can rely on the individual objects’ collision() and canInteract() methods (they’re identical to those in the python simulation); be a bit more wary of the other ones. The helper methods are still very much under active development.
