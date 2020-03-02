import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-white')

plt.xkcd()
_, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)


N = 10000

x, y = np.random.uniform(0, 1, size=(2, N))
inside = (x**2 + y**2)  <= 1

outside = np.invert(inside)

ax[0].plot(x[inside], y[inside], 'C0.')
ax[0].plot(x[outside], y[outside], 'C1.')
ax[0].axis('square')
ax[0].set_xticks([])
ax[0].set_yticks([])


dims = []
prop = []
for d in range(2, 15):
    x = np.random.random(size=(d, N))
    inside = ((x * x).sum(axis=0) < 1).sum()
    dims.append(d)
    prop.append(inside / N)
    
ax[1].plot(dims, prop);
ax[1].set_xticks(dims);
ax[1].set_xlabel('Dimensions', fontsize=14)
ax[1].set_ylabel('Fraction of interior dots', labelpad=20, fontsize=14);
plt.savefig('dimensiones.png')


plt.figure(figsize=(10, 4))
for c, d in enumerate([1, 10, 100, 500]):
    samples = stats.multivariate_normal([0]*d, np.eye(d)).rvs(5000)
    if d == 1:
        radial_dist = np.sum(samples[:, None]**2, 1)**0.5
    else:
        radial_dist = np.sum(samples**2, 1)**0.5
    #az.plot_kde(radial_dist, plot_kwargs={'color':f'C{c}'}, label=f'd = {d}')
    az.plot_kde(radial_dist, plot_kwargs={"alpha":1/(d**0.2)}, label=f'd = {d}')
plt.yticks([])
plt.xlabel('Distance to the mode');
plt.savefig('distancia_a_la_moda.png')
