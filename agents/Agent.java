import java.util.Arrays;
import com.supermarket.*;

// Use this as a template to develop your agent
public class Agent extends SupermarketComponentImpl {

    public Agent() {
	super();
	shouldRunExecutionLoop = true;
	log.info("In Agent constructor.");
    }

    boolean firsttime = true;
    
    @Override
    protected void executionLoop() {
	// this is called every 100ms
	/* put your code in here, you can use the following functions to control your agent:
	   nop();
	   startGame();
	   resetGame();
	   goNorth();
	   goSouth();
	   goEast();
	   goWest();
	   toggleShoppingCart();
	   interactWithObject();
	   cancelInteraction();
	*/
	SupermarketObservation obs = getLastObservation();
	if (firsttime) {
	    System.out.println(obs.players.length + " players");
	    System.out.println(obs.carts.length + " carts");
	    System.out.println(obs.shelves.length + " shelves");
	    System.out.println(obs.counters.length + " counters");
	    System.out.println(obs.registers.length + " registers");
	    System.out.println(obs.cartReturns.length + " cartReturns");
	    // print out the shopping list
	    System.out.println("Shoppping list: " + Arrays.toString(obs.players[0].shopping_list));
	    firsttime = false;
	}
	// get cart
	SupermarketObservation.CartReturn cartreturn = obs.cartReturns[0];
	double x = cartreturn.position[0];
	double y = cartreturn.position[1];
	SupermarketObservation.Player me = obs.players[0];
	double myx = me.position[0];
	double myy = me.position[1];
	System.out.println("Player (x,y)     " + myx + " " + myy);
	System.out.println("CartReturn (x,y) " + x + " " + y);
	goEast();
	goEast();
	goEast();
	while(!cartreturn.canInteract(me)) {
	    goSouth();
	    obs = getLastObservation();
	    cartreturn = obs.cartReturns[0];
	    me = obs.players[0];
	    /*
	    myx = me.position[0];
	    myy = me.position[1];
	    System.out.println("Player (x,y)     " + myx + " " + myy);
	    */
	}
	// pick up the cart
	System.out.println("Picking up cart");
	interactWithObject();
	// keeps going north, bumping into the counter
	while(true) {
	    goNorth();
	}
    }
}
