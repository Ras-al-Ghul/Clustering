import sys
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

mindist = False	#If set to True, mindist is the distance metric, else it is meandist

# HTML ColorCodes
Color = ['#ff2d00', '#000000', '#ff0064', '#00aeff', '#fffb00', \
		 '#aeff00', '#0f00ff', '#00ff64', '#00ff83', '#00ffa2', \
		 '#00ffc9', '#00fbff', '#ff8b00', '#0061ff', '#23ff00', \
		 '#8f00ff', '#ff00fb', '#ff00b6', '#ff4900', '#ff001b']

# For TNEB dataset
Type = ['Bank', 'AutomobileIndustry', 'BpoIndustry', 'CementIndustry', 'Farmers1', \
		'Farmers2', 'HealthCareResources', 'TextileIndustry', 'PoultryIndustry', \
		'Residential(individual)', 'Residential(Apartments)', 'FoodIndustry', \
		'ChemicalIndustry', 'Handlooms', 'FertilizerIndustry', 'Hostel', 'Hospital', \
		'Supermarket', 'Theatre', 'University']
ServiceID = ['671004572', '457008451', '581000256', '775001231', '455007891', '562321452', \
			 '450023897', '785200123', '568730109', '609822556', '894536726', '978045321', \
			 '5783456902', '819034567', '945678934', '486589321', '389457902', '256835671', \
			 '198346752', '286130985', '374897109', '498710889', '693421673', '785643218', \
			 '785643223', '652132542', '450012212', '548542561', '524100231', '600124212', '800145754']

# For 3D Road Network dataset
# Empty as of now

class Node:
	def __init__(self, vec, left = None, right = None, distance = 0.0, ids = None, num = 1):
		self.left = left
		self.right = right
		self.vec = vec
		self.id = ids
		self.num = num	#Num of elements in that cluster
		self.distance = distance

def meandistance(Node1, Node2):
	sums = 0
	for i in range(len(Node1.vec)):
		sums += (Node1.vec[i] - Node2.vec[i]) ** 2
	return sums ** 0.5

def mindistance(vec1, vec2):
	sums = 0
	for i in range(len(vec1)):
		sums += (vec1[i] - vec2[i]) ** 2
	return sums ** 0.5

def mindistclusterize(clusters):
	distcache = {}
	clustid = -1
	
	while len(clusters) > 1:
		lowestmindistpair = (0,1)
		closestmindist = sys.maxint
		
		for i in range(len(clusters)):
			for j in range(i+1, len(clusters)):
				for k in range(len(clusters[i].vec)):
					for l in range(len(clusters[j].vec)):
						hasha = [clusters[i].vec[k][m] for m in range(len(clusters[i].vec[k]))]
						hashaa = [hasha[m - 1] * m for m in range(1, len(hasha) + 1)]
						hashb = [clusters[j].vec[l][m] for m in range(len(clusters[j].vec[l]))]
						hashbb = [hashb[m - 1] * m for m in range(1, len(hashb) + 1)]
						if (sum(hashaa), sum(hashbb)) not in distcache:
							distcache[(sum(hashaa), sum(hashbb))] = mindistance(clusters[i].vec[k], clusters[j].vec[l])

						tempdist = distcache[(sum(hashaa), sum(hashbb))]

						if tempdist < closestmindist:
							closestmindist = tempdist
							lowestmindistpair = (i, j)

		a, b = lowestmindistpair[0], lowestmindistpair[1]
		#Merge
		tempvec = clusters[a].vec + clusters[b].vec

		newNode = Node(vec = tempvec, left = clusters[a], right = clusters[b], distance = closestmindist, \
						ids = clustid, num = (len(clusters[a].vec) + len(clusters[b].vec)))
		clustid -= 1
		del clusters[b]
		del clusters[a]
		clusters.append(newNode)

	return clusters

def meandistclusterize(clusters):
	distcache = {}
	clustid = -1

	while len(clusters) > 1:
		lowestmeandistpair = (0,1)
		closestmeandist = meandistance(clusters[0], clusters[1])

		for i in range(len(clusters)):
			for j in range(i+1, len(clusters)):
				if (clusters[i].id, clusters[j].id) not in distcache:
					distcache[(clusters[i].id, clusters[j].id)] = meandistance(clusters[i], clusters[j])

				tempdist = distcache[(clusters[i].id, clusters[j].id)]

				if tempdist < closestmeandist:
					closestmeandist = tempdist
					lowestmeandistpair = (i, j)

		a, b = lowestmeandistpair[0], lowestmeandistpair[1]
		#Merge
		tempvec = [float((clusters[a].vec[i]) * clusters[a].num + (clusters[b].vec[i]) * clusters[b].num)\
					/(clusters[a].num + clusters[b].num) for i in range(len(clusters[a].vec))]

		newNode = Node(tempvec, left = clusters[a], right = clusters[b], distance = closestmeandist, \
						ids = clustid, num = (clusters[a].num + clusters[b].num))

		clustid -= 1
		del clusters[b]
		del clusters[a]
		clusters.append(newNode)

	return clusters

def clusterize(data):
	if not mindist:
		clusters = [Node(data[i], ids = i) for i in range(len(data))]
		return meandistclusterize(clusters)
	else:
		clusters = [Node(vec = [data[i]], ids = i) for i in range(len(data))]
		return mindistclusterize(clusters)

def createDataandVisualize():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels = make_blobs(n_samples = 90, centers = centers, cluster_std = 0.4)

    for i in range(len(set(labels))):
    	x, y = [], []
    	for j in range(len(labels)):
    		if labels[j] == i:
    			x.append(X[j][0])
    			y.append(X[j][1])
    	plt.scatter(x, y, color=Color[i])

    #plt.show()
    return X, labels

def visualizeTNEBdataset():
	TNEBcountsType = [0 for x in Type]
	TNEBcountsServiceID = [0 for x in ServiceID]

	Typewise = [[] for x in Type]

	newdata = []

	fp = open('electricityboarddata.txt', 'r')
	data = fp.readlines()

	for i in range(len(data)):
		line = data[i]
		line = line.split(',')
		line[len(line) - 1] = line[len(line) - 1][:len(line[len(line) - 1]) - 1]
		
		TNEBcountsType[Type.index(line[2])] += 1
		TNEBcountsServiceID[ServiceID.index(line[4])] += 1

		if TNEBcountsType[Type.index(line[2])] < 10:
			Typewise[Type.index(line[2])].append([float(line[0]), float(line[1])])
			newdata.append([float(line[0]), float(line[1])])

	for i in range(len(Typewise)):
		x, y = [], []
		for j in range(len(Typewise[i])):
			x.append(Typewise[i][j][0])
			y.append(Typewise[i][j][1])
		plt.scatter(x, y, color=Color[i])

	plt.show()

	print TNEBcountsType
	print TNEBcountsServiceID
	return newdata, []


def getClusters(root, dist):
	if root.distance < dist:
		return [root]
	left = []
	right = []
	if root.left != None:
		left = getClusters(root.left, dist = dist)
	if root.right != None:
		right = getClusters(root.right, dist = dist)
	return left + right

def getElements(root):
	if root.id >= 0:
		return [root.id]
	left = []
	right = []
	if root.left != None:
		left = getElements(root.left)
	if root.right != None:
		right = getElements(root.right)
	return left + right

def displayClustering(data, clusters):
	print len(clusters)
	for i in range(len(clusters)):
		tempindices = getElements(clusters[i])
		arrofvecs = [data[j] for j in range(len(data)) if j in tempindices]
		xs = [arrofvecs[j][0] for j in range(len(arrofvecs))]
		ys = [arrofvecs[j][1] for j in range(len(arrofvecs))]
		plt.scatter(xs, ys, color=Color[i])

	plt.show()

if __name__ == '__main__':
	# data, actualLabels = visualizeTNEBdataset()
	data, actualLabels = createDataandVisualize()

	root = clusterize(data)[0]
	if not mindist:
		clusters = getClusters(root, dist = 0.9)
	else:
		clusters = getClusters(root, dist = 0.6)
	
	displayClustering(data, clusters)
	

