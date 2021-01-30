import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from calc import Computer

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)

labels = ['Prec Hate', 'Rec Hate', 'F1 Hate', 'Prec Off', 'Rec Off', 'F1 Off', 'F1 Macro']
x = np.arange(len(labels))

g0 = 0.16
hg0 = 0.8
hi0 = 0.4
ch0 = 0.5
co0 = 0.8

c = Computer(g0, hg0, hi0, ch0, co0)
values = c.values_from_labels(labels)
width = 0.2

bars = ax.bar(x, values, width, color='r', tick_label=labels)
ax.set_ylabel('Percentage')
ax.set_yticks(np.arange(0.0, 1.0, 0.1))


axcolor = 'lightgoldenrodyellow'
ax_g = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_hg = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_hi = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_ch = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_co = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)

s_g = Slider(ax_g, 'P(Grp)', 0.0, 1.0, valinit=c._init_probs['g'])
s_hg = Slider(ax_hg, 'P(Hate|Grp)', 0.0, 1.0, valinit=c._init_probs['h_g'])
s_hi = Slider(ax_hi, 'P(Hate|Ind)', 0.0, 1.0, valinit=c._init_probs['h_i'])
s_ch = Slider(ax_ch, 'P(Correct|Hate)', 0.0, 1.0, valinit=c._init_probs['c_h'])
s_co = Slider(ax_co, 'P(Correct|Off)', 0.0, 1.0, valinit=c._init_probs['c_o'])



def update(val):
  c.update(s_g.val, s_hg.val, s_hi.val, s_ch.val, s_co.val)
  values = c.values_from_labels(labels)
  for rect, value in zip(bars, values):
    rect.set_height(value)
  fig.canvas.draw_idle()



s_g.on_changed(update)
s_hg.on_changed(update)
s_hi.on_changed(update)
s_ch.on_changed(update)
s_co.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
  s_g.reset()
  s_hg.reset()
  s_hi.reset()
  s_ch.reset()
  s_co.reset()

button.on_clicked(reset)

"""
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)
"""

plt.show()