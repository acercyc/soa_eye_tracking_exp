
## 🧠 **Adaptive Gaze Classification via Temporal Embedding and Clustering**

### **Overview**

This algorithm classifies eye movements into **fixations** and **saccades** using a fully  **unsupervised** ,  **geometry-aware** , and **adaptive** method. It does not rely on hand-tuned thresholds (e.g., velocity) but instead discovers the natural structure of gaze dynamics via clustering in a  **temporal embedding space** .

---

### **How It Works**

#### **Stage 1: Temporal Embedding and Clustering (Initialization)**

* For each time point tt, construct a **6D embedding vector** using the gaze positions at times t−2,t−1,tt-2, t-1, t:

  xt=[xt−2,xt−1,xt,yt−2,yt−1,yt]\mathbf{x}_t = [x_{t-2}, x_{t-1}, x_t, y_{t-2}, y_{t-1}, y_t]
* Apply **Gaussian Mixture Modeling (GMM)** with n=2n = 2 on an initial batch of embeddings.
* Automatically label the clusters:

  * The **compact, low-variance** cluster is identified as  *fixation* .
  * The **larger, more variable** cluster is identified as  *saccade* .

#### **Stage 2: Real-Time Classification**

* New samples are classified by computing their **Mahalanobis distance** to each cluster’s centroid and covariance.
* The sample is assigned to the closest cluster, yielding a  **fixation/saccade decision** .

#### **Stage 3: Adaptive Updating**

* Cluster statistics (mean and covariance) are **incrementally updated** using exponential moving averages as new, high-confidence samples arrive.
* This allows the algorithm to **adapt over time** to drift in gaze behavior or changes in task/environment.

---

### **Key Advantages**

* ✅ **Fully data-driven** — no velocity or dispersion thresholds
* ✅ **Robust to noise and drift**
* ✅ **Geometrically informed** — captures temporal dynamics over 3 points
* ✅ **Efficient** — prediction uses only centroids and covariance

```
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

class GazeSaccadeFixationClassifier:
    def __init__(self, embedding_window=3, init_samples=500, n_components=2):
        self.embedding_window = embedding_window
        self.init_samples = init_samples
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.gmm = None
        self.cluster_stats = {}
        self.fitted = False
        self.label_map = {}  # maps cluster index to 'fixation' or 'saccade'

    def _create_embedding(self, gaze_x, gaze_y):
        """Create temporal embedding for a list of gaze_x and gaze_y values."""
        embeddings = []
        for i in range(self.embedding_window - 1, len(gaze_x)):
            x_vals = gaze_x[i - 2:i + 1]
            y_vals = gaze_y[i - 2:i + 1]
            embeddings.append(np.concatenate([x_vals, y_vals]))
        return np.array(embeddings)

    def initialize(self, gaze_x, gaze_y):
        """Stage 1: Initialize with a batch of data to fit initial clusters."""
        embeddings = self._create_embedding(gaze_x, gaze_y)
        embeddings_std = self.scaler.fit_transform(embeddings)
  
        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        self.gmm.fit(embeddings_std)
  
        # Determine which cluster is fixation based on smallest average Mahalanobis distance
        mahal_dists = []
        for i in range(self.n_components):
            mu = self.gmm.means_[i]
            cov = self.gmm.covariances_[i]
            inv_cov = np.linalg.inv(cov)
            dists = [mahalanobis(x, mu, inv_cov) for x in embeddings_std[self.gmm.predict(embeddings_std) == i]]
            mahal_dists.append(np.mean(dists))
  
        fixation_cluster = np.argmin(mahal_dists)
        saccade_cluster = 1 - fixation_cluster
        self.label_map = {fixation_cluster: 'fixation', saccade_cluster: 'saccade'}

        # Store initial stats
        self.cluster_stats = {
            i: {
                'mean': self.gmm.means_[i],
                'cov': self.gmm.covariances_[i],
                'inv_cov': np.linalg.inv(self.gmm.covariances_[i]),
                'count': 1
            } for i in range(self.n_components)
        }
        self.fitted = True

    def classify(self, gaze_x_window, gaze_y_window):
        """Stage 2: Classify a new sample based on nearest cluster."""
        if not self.fitted:
            raise RuntimeError("Model has not been initialized.")
        x = np.concatenate([gaze_x_window, gaze_y_window])
        x_std = self.scaler.transform([x])[0]
  
        dists = {
            i: mahalanobis(x_std, self.cluster_stats[i]['mean'], self.cluster_stats[i]['inv_cov'])
            for i in self.cluster_stats
        }
        best_cluster = min(dists, key=dists.get)
        return self.label_map[best_cluster], best_cluster, dists[best_cluster]

    def update(self, x, y, label_idx, alpha=0.05):
        """Stage 3: Adaptively update cluster stats using exponential moving average."""
        x_embed = np.concatenate([x, y])
        x_std = self.scaler.transform([x_embed])[0]

        cluster = self.cluster_stats[label_idx]
        # Exponential Moving Average
        cluster['mean'] = (1 - alpha) * cluster['mean'] + alpha * x_std
        cluster['cov'] = (1 - alpha) * cluster['cov'] + alpha * np.outer(x_std - cluster['mean'], x_std - cluster['mean'])
        cluster['inv_cov'] = np.linalg.inv(cluster['cov'])
        cluster['count'] += 1

    def predict_and_update(self, gaze_x_window, gaze_y_window, confidence_threshold=None):
        """Full classification + update loop."""
        label, cluster_idx, dist = self.classify(gaze_x_window, gaze_y_window)

        # Optional confidence-based gating
        if confidence_threshold is None or dist < confidence_threshold:
            self.update(gaze_x_window, gaze_y_window, cluster_idx)

        return label, dist

```
