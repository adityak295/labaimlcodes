MIN = -1000
MAX = 1000
def minmax(depth, node, values, isMax, alpha, beta, d):
    if depth==d:
        return values[node]
    best = MIN if isMax else MAX
    for i in range(2):
        val = minmax(depth+1, node*2+i, values, not isMax, alpha, beta, d)
        if isMax:
            best = max(best,val)
            alpha = max(best,alpha)
        else:
            best = min(best,val)
            beta = min(beta,best)
        if beta<=alpha:
            break
    return best
values = [3,5,6,9,1,2,0,-1]
print("The optimal value is: ", minmax(0,0,values,True,MIN,MAX,3))
