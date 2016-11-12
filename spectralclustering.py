import sys, random, math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

Epsilonneighbourhood = True #Toggle for Epsilon neighbourhood similarity graph vs Gaussian similarity function graph
numofclusters = 3

# HTML ColorCodes
Color = ['#ff2d00', '#000000', '#fffb00', '#ff0064', '#00aeff', \
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

def createDataandVisualize():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels = make_blobs(n_samples = 400, centers = centers, cluster_std = 0.4)

    for i in range(len(set(labels))):
    	x, y = [], []
    	for j in range(len(labels)):
    		if labels[j] == i:
    			x.append(X[j][0])
    			y.append(X[j][1])
    	plt.scatter(x, y, color=Color[i])

    # plt.show()
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

def calcgaussiansimilarity(vec1, vec2):
	gamma = 10
	return math.exp(-(np.linalg.norm(vec1 - vec2) ** 2) * gamma)

def calceneighbourhood(vec1, vec2):
	epsilon = 2
	dist = np.linalg.norm(vec1 - vec2)
	if dist <= epsilon:
		return 1
	else:
		return 0

def constructSimilarityMatrix(data, numofdata, labels):
	similaritymeasure = calcgaussiansimilarity
	if Epsilonneighbourhood:
		similaritymeasure = calceneighbourhood

	similaritymatrix = np.zeros((numofdata, numofdata))

	for i in range(numofdata):
		for j in range(i, numofdata):
			dist = similaritymeasure(data[i], data[j])

			similaritymatrix[i][j] = dist
			similaritymatrix[j][i] = dist

	# for i in range(numofdata):
	# 	for j in range(i, numofdata):
	# 		if labels[i] == labels[j]:
	# 			similaritymatrix[i][j] = 1
	# 			similaritymatrix[j][i] = 1
	# 		else:
	# 			similaritymatrix[i][j] = 0
	# 			similaritymatrix[j][i] = 0

	return similaritymatrix

def getUnnormalizedLaplacian(similaritymatrix, numofdata):
	degrees = [sum(similaritymatrix[i]) for i in range(similaritymatrix.shape[0])]

	degreematrix = np.zeros((numofdata, numofdata))

	for i in range(numofdata):
		degreematrix[i][i] = degrees[i]

	return (degreematrix - similaritymatrix)

def deletecolumns(eigvecs, numofdata):
	for i in range(numofdata - numofclusters):
		eigvecs = np.delete(eigvecs, len(eigvecs[0]) - 1, axis=1)

	return eigvecs

def solveKMeans(eigvecs, numofdata, data):
	datadict = {}
	neweigvecs = np.zeros((numofdata, numofclusters + 1))
	# 0, 1, ...n-1, nth index is cluster
	for i in range(numofdata):
		temp = list(eigvecs[i])
		temp.append(random.randrange(0, numofclusters))
		temp = np.array(temp)
		neweigvecs[i] = temp

	clusters = [[] for i in range(numofclusters)]
	for i in range(numofdata):
		temp = list(neweigvecs[i])
		temp.append(i)
		datadict[i] = temp
		clusters[int(temp[len(temp) - 2])].append(temp)

	flag = False
	iters = 0

	while  not flag:
		phim = [[float(sys.maxint), -1] for i in range(numofdata)]
		flag = True
		templist = []	#To be deleted from original cluster
		lenofclusters = [len(clusters[i]) for i in range(numofclusters)]
		
		clustermeans = [np.zeros((1, numofclusters)) for i in range(numofclusters)]
		
		for i in range(numofclusters):
			temps = [0 for j in range(numofclusters)]
			for j in range(len(clusters[i])):
				for k in range(numofclusters):
					temps[k] += clusters[i][j][k]
			for k in range(numofclusters):
				try:
					temps[k] = float(temps[k])/len(clusters[i])
				except:	#Stuck at local minima, start afresh
					solveKMeans(eigvecs, numofdata, data)
					return
			clustermeans[i] = np.array(temps)

		for i in range(numofclusters):
			for j in range(lenofclusters[i]):
				for k in range(numofclusters):
					tempfullvec = list(clusters[i][j])
					ids = tempfullvec.pop()
					clusternum = tempfullvec.pop()
					tempfullvec = np.array(tempfullvec)

					clustermean = clustermeans[k]

					sqdist = np.linalg.norm(tempfullvec - clustermean) ** 2

					if sqdist < phim[ids][0]:
						flag = False
						phim[ids][0] = sqdist
						phim[ids][1] = k
						if [i, clusters[i][j]] not in templist:
							templist.append([i, clusters[i][j]])

		# Remove changed
		for i in range(len(templist)):
			clusters[templist[i][0]].remove(templist[i][1])
		# Add changed
		for i in range(numofdata):
			if phim[i][1] != -1:
				tempvec = datadict[i]
				tempvec[len(tempvec) - 2] = phim[i][1]
				clusters[phim[i][1]].append(tempvec)

		iters += 1
		if iters == 10:
			flag = True

	lenofclusters = [len(clusters[i]) for i in range(numofclusters)]
	print lenofclusters
	for i in range(len(clusters)):
		x, y = [], []
		for j in range(len(clusters[i])):
			index = clusters[i][j][len(clusters[i][j]) - 1]
			x.append(data[index][0])
			y.append(data[index][1])
		plt.scatter(x, y, color=Color[i])

	plt.show()

if __name__ == '__main__':
	random.seed()
	# data, actualLabels = visualizeTNEBdataset()
	data, actualLabels = createDataandVisualize()
	numofdata = len(data)

	similaritymatrix = constructSimilarityMatrix(data, numofdata, actualLabels)

	Laplacian = getUnnormalizedLaplacian(similaritymatrix, numofdata)

	eigvals, eigvecs = np.linalg.eigh(Laplacian)
	eigvecs = deletecolumns(eigvecs, numofdata)

	solveKMeans(eigvecs, numofdata, data)