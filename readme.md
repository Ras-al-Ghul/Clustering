# Clustering

**To run:** `python filename.py`  

Note that ideally, one performs clustering on real world datasets. This was the objective here too. And hence two datasets were downloaded - The `Tamilnadu Electricity Board Hourly Readings Data Set` <https://archive.ics.uci.edu/ml/datasets/Tamilnadu+Electricity+Board+Hourly+Readings> and the `3D Road Network (North Jutland, Denmark) Data Set` <https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29> Both have few features and hence can be plotted and visualized. But the problem was that there were no clear cut clusters based on the class labels. Another problem was that there was a large number of class labels. Hence for understanding clustering algorithms, I decided to create `blobs` using scikit-learn and then clusterize them - each blob as a cluster - ideally.  

1. **Agglomerative Clustering :** This implements two cluster merging criterion:  
  - `min distance` criterion - Merge the two clusters which have the minimum distance between two points - one from each cluster.  
  - `mean distance` criterion - Merge two clusters based on the mean of all the points in each cluster.  
Set the `mindist` flag in line 5. Also take care about the `dist` parameter in lines 220 and 222. Changing them gives different number of clusters.  
Reference used <http://www.janeriksolem.net/2009/04/hierarchical-clustering-in-python.html>  
2. **Kernel KMeans Clustering :** Two kernels have been used here:  
  - `polynomial kernel`  
  - `RBF kernel`  
Set the `RBFkernel` flag in line 6 and the desired number of clusters as `numofclusters` in line 7. One can tune the params of the Polynomial and the RBF kernel in the methods `Poly` and `RBF`.  
Reference used <http://www.cs.ucsb.edu/~veronika/MAE/Global_Kernel_K-Means.pdf>  
3. **Spectral Clustering :** Two similarity measures have been implemented here for constructing the similarity matrix:  
  - `Gaussian similarity`  
  - `Epsilon Neighbourhood similarity`  
Set the `Epsilonneighbourhood` flag in line 6 and the desired number of clusters as `numofclusters` in line 7. Be very careful to set the `gamma` in `calcgaussiansimilarity` and the `epsilon` in the `calceneighbourhood` methods as this clustering algorithm is very sensitive to how we define the similarity measure. If unsure, uncomment lines 107 to 114 and comment out lines 100 to 105 - sets the similarity matrix according to the labels - which is the ideal case. Another thing to note is that KMeans can get stuck at local minima - see line 170  
Reference used <http://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf>  
