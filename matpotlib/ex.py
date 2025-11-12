import matplotlib.pyplot as plt
import numpy as np

sehwag = [0, 300, 800, 1200, 1500, 1700, 1600, 1400, 1000, 0]
years = [1990, 1992, 1994, 1996, 1998, 2000, 2003, 2005, 2007, 2010] 
kohli =  [500, 700, 1100, 1500, 1800, 1200, 1700, 1300, 900, 1500] 
# with plt.xkcd():
#     plt.plot(years, kohli, label = "virat kohli")
#     plt.plot(years, sehwag, label = "apna bhai")
#     plt.legend()

players = ["pradeep", "manji", "pawan", "Shardul", "fazal"]
raids = [2000, 1500, 1700, 1000-500, 200] 

plt.pie(raids, labels=players)
plt.title("top player")
plt.show()