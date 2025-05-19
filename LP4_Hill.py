import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return -x**2

def hill_climbing(intial_x=1,max_iterations=1000,step_size=0.9):
  current_x=intial_x
  x_values=[intial_x]
  current_y=f(intial_x)
  y_values=[current_y]

  for i in range(max_iterations):
    neighbour_x=current_x + step_size * np.random.uniform(-1,1)
    neighbout_y=f(neighbour_x)

    if neighbout_y > current_y:
      current_x = neighbour_x
      current_y = neighbout_y

      x_values.append(current_x)
      y_values.append(current_y)

  return current_x,current_y,x_values,y_values

def plotHillClimb(x_values,y_values):
  graph_x=np.linspace(min(x_values),max(x_values))
  graph_y=[f(g_x) for g_x in graph_x]

  plt.plot(graph_x,graph_y)
  plt.plot(x_values,y_values)
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.title('Hill Climbing Plot')

results=hill_climbing()
print(results)
plotHillClimb(results[2],results[3])
