>  This file contains the approaches taken for this code and why the next approach was taken
> in underlying the bad thinking in the previous approach

### The approach for checking obstacles
The underlying idea was to check if there was a wall or a crate near the immediate
vicinity of the agent and if the agent makes an invalid move in the next round of the game
then penalize the agent else rewarding it for making a correct move.

The idea was not successful as for creating an initial Q-table we didn't have enough number of states
for our agent to learn anything in the process. Also, the idea itself was not a good call in long term 
because agent also needed to distinguish between the wall and crate.

#### Update on the indexing logic
The indexing logic in this case is bad as well because the higher number of states can not be maintained with
a simple index.

### New approach for checking wall and creating a feature dictionary
The idea technically remains the same with me checking the walls(only) near an agent and telling it to move in 
a direction which is not an obstacle

so in this case the valid states can look like [UP, DOWN, LEFT, RIGHT, OBSTACLE] and here I can penalize the agent 
if the agent has chosen an obstacle with a 'BAD_MOVE' event. I will further make a list of features having all the 
possible combinations of the states.

The addition here is that to increase the number of states, I will be adding a all direction move, so that our agent
knows that upon moving to that direction it will get rewarded.

### Approach for finding coins
The idea is to find a coin that is near to the agent and then steer the agent towards it. I see if there are any coins
near the agent within the radius of 5 tiles(if the game is 17*17, it is increased while training if the field is larger)
and when I found the fist coin I break the coin finding loop as that will give me only the coin nearest to the agent.

Once I have the coin location I calculate the nearest path to the coin and get the most immediate action that our agent
should take to get the coin and then reward the agent based on that. Also, In case I find no coin near the agent, for now
I just take a random action.

#### path to nearest coin
There are multiple algorithms that can be used to find a path to the nearest element, but for the ease of use and simplicity,
I will be implementing Bi-directional BFS to find the path between agent and coin which will also reduce the search space.

The problem with this strategy was that the Q-table was excelling in a negative reward systems so reward tuning and additional
events is necessary.