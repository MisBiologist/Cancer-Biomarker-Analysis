import seaborn as sns

# clustermap automatically reorders rows/cols by similarity
g = sns.clustermap(
    corr,
    cmap=cmap,
    linewidths=.5,
    figsize=(16, 16),
    annot=False,          # skip annotations here for clarity
    cbar_kws={"shrink": .5}
)
g.fig.suptitle('Clustered Correlation Map', y=1.02, fontsize=18)
g.savefig('clustermap.png', dpi=300)
plt.show()