import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img=plt.imread(r"C:\Users\navan\Downloads\1.jpeg")/255
print(img.shape)
plt.imshow(img)
plt.show()

img_reshape = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
print(img_reshape.shape)

kmeans = KMeans(n_clusters=5, random_state=0).fit(img_reshape)
img2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = img2show.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(cluster_pic)
plt.show()

'''
Clustering is dividing the population (data points) into many groups, such that data points in the same groups are more similar to other data points in that group than those in other groups. These groups are known as clusters.
'''
