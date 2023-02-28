#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
data = np.random.randint(0, 20, (4,3))
x = ('Farrah', 'Fred', 'Felicia')
colors = ('red','yellow','#ff8000','#ffe5b4')
bar_width = 0.5
n_rows = len(data)
index = np.arange(len(columns)) + 0.3

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(x, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    

plt.ylabel("Quantity of Fruit")
plt.legend(['red','yellow','orange','peach'])
plt.title("Number of Fruit per Person")
plt.ylim(0,80)
plt.show()
