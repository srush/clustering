import matplotlib
import matplotlib.pyplot as plt

def draw_points(points):
  colors = ["red", "yellow", "green", "blue", "pink", "purple", "orange", "brown", "white"]
  plt.xticks([])
  plt.yticks([])
  plt.xlim([0, 1.0])
  plt.ylim([0, 1.0])
  
  for phoneme, points in points.iteritems():
    n = phoneme.id % len(colors)
    for p in points:
      x = [p.point[0]]
      y = [p.point[1]]
      plt.scatter(x, y, c=colors[n])      
  plt.show()
