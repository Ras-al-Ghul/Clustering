import sys, random, math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

RBFkernel = False #Toggle for RBFkernel and Polynomialkernel
numofclusters = 3

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

def createClusters(data, labels):
	datadict = {}
	clusters = [[] for i in range(numofclusters)]
	for i in range(len(data)):
		tempvec = list(data[i])
		tempvec.append(i)	#id
		#clusters[random.randrange(numofclusters)].append(tempvec)
		clusters[labels[i]].append(tempvec)
		datadict[i] = tempvec
	return clusters, datadict

def Poly(vec1, vec2):
	degree = 2
	gamma = 0.5
	newvec1, newvec2 = vec1[:len(vec1) - 1], vec2[:len(vec2) - 1]
	newvec1, newvec2 = np.array(newvec1), np.array(newvec2)
	return ((np.dot(newvec1, newvec2) + gamma) ** degree)

def RBF(vec1, vec2):
	gamma = 5
	newvec1, newvec2 = vec1[:len(vec1) - 1], vec2[:len(vec2) - 1]
	newvec1, newvec2 = np.array(newvec1), np.array(newvec2)
	return math.exp(-(np.linalg.norm(newvec1 - newvec2) ** 2) * gamma)

def kernelKMeans(clusters, numofdata, datadict):
	kernel = Poly
	if RBFkernel:
		kernel = RBF
	flag = False
	iters = 0
	while not flag:
		phim = [[float(sys.maxint), -1] for i in range(numofdata)]
		flag = True
		templist = []	#To be deleted from original cluster
		lenofclusters = [len(clusters[i]) for i in range(numofclusters)]
		for i in range(numofclusters):
			for j in range(lenofclusters[i]):
				for k in range(numofclusters):
					firstterm = kernel(clusters[i][j], clusters[i][j])
					
					secondterm = float(0)
					for l in range(lenofclusters[k]):
						secondterm += kernel(clusters[i][j], clusters[k][l])
					secondterm *= (-2)
					secondterm /= lenofclusters[k]
					
					thirdterm = float(0)
					for l in range(lenofclusters[k]):
						for m in range(lenofclusters[k]):
							thirdterm += kernel(clusters[k][l], clusters[k][m])
					thirdterm /= (lenofclusters[k] ** 2)

					term = firstterm + secondterm + thirdterm
					if term < phim[clusters[i][j][2]][0]:
						phim[clusters[i][j][2]][0] = term
						phim[clusters[i][j][2]][1] = k
						flag = False
						if [i, clusters[i][j]] not in templist:
							templist.append([i, clusters[i][j]])
		
		# Remove changed
		for i in range(len(templist)):
			clusters[templist[i][0]].remove(templist[i][1])
		# Add changed
		for i in range(numofdata):
			if phim[i][1] != -1:
				tempvec = datadict[i]
				clusters[phim[i][1]].append(tempvec)

		iters += 1
		if iters == 10:
			flag = True

	for i in range(len(clusters)):
		x, y = [], []
		for j in range(len(clusters[i])):
			x.append(clusters[i][j][0])
			y.append(clusters[i][j][1])
		plt.scatter(x, y, color=Color[i])

	plt.show()

if __name__ == '__main__':
	random.seed()
	# data, actualLabels = visualizeTNEBdataset()
	data, actualLabels = createDataandVisualize()
	numofdata = len(data)

	clusters, datadict = createClusters(data, actualLabels)
	# print len(clusters[0]), len(clusters[1]), len(clusters[2])
	kernelKMeans(clusters, numofdata, datadict)