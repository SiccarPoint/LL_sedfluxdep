from matplotlib.pyplot import plot, show, ylim
from landlab import RasterModelGrid
from landlab.components import SedDepEroder
import numpy as np
import seaborn as sns

# for fig 1
mycolors = sns.color_palette("cubehelix", 5)
lines = ["-","--","-.",":"]

mg = RasterModelGrid((50, 50), 200.)
z = mg.add_zeros('node', 'topographic__elevation')
sdegen = SedDepEroder(mg)
# clear the output fields to permit binding of other components
for field in sdegen.output_var_names[1:]:
    mg.at_node.pop(field)
sdepara = SedDepEroder(mg, sed_dependency_type='almost_parabolic')
for field in sdegen.output_var_names[1:]:
    mg.at_node.pop(field)
sdelindec = SedDepEroder(mg, sed_dependency_type='linear_decline')
for field in sdegen.output_var_names[1:]:
    mg.at_node.pop(field)
sdeconst = SedDepEroder(mg, sed_dependency_type='None')

sns.set_context('paper')
for ln, col, sde in zip(lines, mycolors[:4], (sdegen, sdepara, sdelindec, sdeconst)):
    with sns.axes_style('whitegrid', {'grid.linestyle': '--'}):
        sde.show_sed_flux_function(color=col, linestyle=ln)
ylim(0, 1.02)
show()
