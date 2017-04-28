import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_query_in_reference(query_data, 
                            reference_data, 
                            model,
                            reference_label='Reference',
                            query_label='Query'):

    tsne = TSNE(n_components=2)
    all_data = np.vstack([model.transform(reference_data), 
                          model.transform(query_data)])

    all_data_tx = tsne.fit_transform(all_data)

    reference_tx = all_data_tx[:reference_data.shape[0], :]
    query_tx = all_data_tx[reference_data.shape[0]:, :]

    plt.scatter(reference_tx[:, 0], query_tx[:, 0], label=reference_label)
    plt.scatter(query_tx[:, 0], query_tx[:, 0], label=query_label)
    plt.legend()
