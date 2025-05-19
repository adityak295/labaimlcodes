def printState(state):
    for row in state:
        print(('|').join(map(str,row)))
    print()

def returnBlank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j]==0:
                return i,j
            
def move(state,direction):
    i,j = returnBlank(state)
    new_state = [row[:] for row in state]
    if direction=='up' and i>0:
        new_state[i-1][j], new_state[i][j] = new_state[i][j], new_state[i-1][j]
        return new_state
    elif direction=='right' and j<2:
        new_state[i][j+1], new_state[i][j] = new_state[i][j], new_state[i][j+1]
        return new_state
    elif direction=='left' and j>0:
        new_state[i][j-1], new_state[i][j] = new_state[i][j], new_state[i][j-1]
        return new_state
    elif direction=='down' and i<2:
        new_state[i+1][j], new_state[i][j] = new_state[i][j], new_state[i+1][j]
        return new_state
    return None

def heuristic(state, goal_state):
    return sum(1 for i in range(3) for j in range(3) if state[i][j]!=goal_state[i][j])

def aStar(initial_state, goal_state):
    open = [[heuristic(initial_state, goal_state), 0, initial_state]]
    closed = []
    while open:
        open.sort()
        f,g,current_state = open.pop(0)
        printState(current_state)
        if current_state==goal_state:
            print("Goal achieved")
            break
        closed.append(current_state)
        for direction in ['up','down','left','right']:
            successor = move(current_state,direction)
            if successor and successor not in closed:
                h = heuristic(successor, goal_state)
                open.append([g+1+h,g+1,successor])
        print("Goal not achieved")

initial_state = [[1,2,3],
                 [8,0,4],
                 [7,6,5]]

goal_state = [[1,2,0],
              [8,6,3],
              [7,5,4]]

aStar(initial_state, goal_state)
