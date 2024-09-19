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