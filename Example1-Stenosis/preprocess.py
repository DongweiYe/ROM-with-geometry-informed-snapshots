import numpy as np
import matplotlib.pyplot as plt
import os
from pyDOE import *
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
import shutil


# def checkIfDuplicates_1(listOfElems):
#     ''' Check if given list contains any duplicates '''
#     if len(listOfElems) == len(set(listOfElems)):
#         return False
#     else:
#         return True


def manual_rbf(Coeffi,CPs,PredInput,shift,scale):
	shift = shift
	scale = scale

	numCP = CPs.shape[0]
	predlist = np.empty((0,2))
	for i in range(PredInput.shape[0]):
		xydis = PredInput[i,:] - CPs
		dis = np.sqrt(np.square(xydis[:,0])+np.square(xydis[:,1]))
		kerlist = np.power(dis,3)

		scalePredInput = (PredInput[i,:] - shift)/scale

		xpred = np.sum(kerlist*Coeffi[:numCP,0])+Coeffi[-3,0]+np.sum(Coeffi[-2:,0]*scalePredInput)
		ypred = np.sum(kerlist*Coeffi[:numCP,1])+Coeffi[-3,1]+np.sum(Coeffi[-2:,1]*scalePredInput)
		pred = np.array([[xpred,ypred]])
		predlist = np.vstack((predlist,pred))
	
	return predlist


def readVTK(filename):
	### Open the vtk data
	Data = open(filename, "r+")
	text = Data.readlines()

	### Create a list to store the data and remove first 5 head text line
	BoundaryList = []
	text = text[5:]

	### For loop all lines and get information of points
	for i in range(len(text)):
		currentline = text[i]
		coordinates = currentline.strip().split()
		x = np.float64(coordinates[0])
		y = np.float64(coordinates[1])
		z = np.float64(coordinates[2])
		BoundaryList.append([x,y,z])
	BoundaryList = np.asarray(BoundaryList)

	### Check if it is a 2D/3D data. If 2D drop last column.
	if np.sum(BoundaryList[:,2]) == 0:
		return BoundaryList[:,:2]
	else:
		return BoundaryList

### This is a manual function to corrent the small deformation on the inlet boundary 
def cleanBC(BoundaryPoints, ReferencePoints):
	for i in range(ReferencePoints.shape[0]):
		if ReferencePoints[i,2] == 4:
			BoundaryPoints[i,:] = ReferencePoints[i,:2]
		elif ReferencePoints[i,2] == 1 and ReferencePoints[i,0] < 1:
			BoundaryPoints[i,:] = ReferencePoints[i,:2]
		elif ReferencePoints[i,2] == 3 and ReferencePoints[i,0] < 1:
			BoundaryPoints[i,:] = ReferencePoints[i,:2]

	return BoundaryPoints


########################################################################################
### This is a data generation and preprocessing files including following process 
###         1) Generate samples (mean and variance for Gaussian function)         
###         2) Shape and mesh generation
###         3) Fetch boundary vertice of meshes 
###         4) Surface registration (using deformetrica)
###         5) Compute the mapping using RBFI and save the data for FEM simulation
###			6) Parametrize the geometry and compute reduced parameters of geometry
########################################################################################


#################################################################################
### Step 1: Generate 500 LHC samples
#################################################################################  
num_samples = 500
A = lhs(4, samples=num_samples)

### Scale each random variables to its orginal ranges
### For 2D Stenosis, mean and std for upper and lower boundaries
A[:,0:1] = A[:,0:1]*(0.7-0.3)+0.3
A[:,1:2] = A[:,1:2]*(0.7-0.3)+0.3 
A[:,2:3] = A[:,2:3]*(2-0.8)+0.8
A[:,3:4] = A[:,3:4]*(2-0.8)+0.8

### Create data folder
try:
	os.makedirs('data/raw/')
except:
	print('Sth went wrong, please check')

### save samples them to the data directory
for sample in range(A.shape[0]):		
	try:
		os.mkdir('data/raw/sample_'+str(sample))
		np.savetxt('data/raw/sample_'+str(sample)+ '/sample.txt',A[sample,:],delimiter=",")
	except:
		print('Sth went wrong, please check')

print('Step 1 - Sample generation: done ')


#################################################################################
### Step 2: Use Freefem to generate meshes for each samples
#################################################################################
os.system("FreeFem++ MeshGeneration.edp -v 0")
print('Step 2 - Mesh generation: done ')


#################################################################################
### Step 3: Fetch vertices from meshes for registration, loop over 500 meshes
#################################################################################
for sample in range(num_samples):
    if sample%100==0:
          print('    Handling sample:',sample)
    
    ### Open up necessary meshes
    TargetMSH = open("data/raw/sample_"+str(sample)+"/sample.msh", "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    t_num = int(words[1])
    e_num = int(words[2])

    TargetMSH.close()
    # print('TOTAL VERTICES:',v_num)

    ### Create a list and copy out all the boundary vertices
    BoundaryList = []
    InnerList = []
    for i in range(1,v_num+1):
        with open("data/raw/sample_"+str(sample)+"/sample.msh",'r') as txt:

            text = txt.readlines()
            currentline = text[i]

            coordinates = currentline.strip().split()
            x = np.round(np.float64(coordinates[0]),decimals=12)
            y = np.round(np.float64(coordinates[1]),decimals=12)
            label =  int(coordinates[2])

            if label != 0:
                BoundaryList.append([x,y,label])
            else:
                InnerList.append([x,y,label])    
    
    BoundaryList = np.asarray(BoundaryList)
    InnerList = np.asarray(InnerList)
    # print(InnerList.shape[0]+BoundaryList.shape[0])

    ### Savem numpy data for boundary vertice list and inner vertice list
    np.save('data/raw/sample_'+str(sample)+'/inner.npy',InnerList)
    np.save('data/raw/sample_'+str(sample)+'/boundary.npy',BoundaryList)

    ## Write a VTK files
    ## Head content: 
    ##     # vtk DataFile Version 2.0
    ##     rawdata, Created by Gmsh
    ##     ASCII
    ##     DATASET POLYDATA
    ##     POINTS 2774 double    
    text_file = open("data/raw/sample_"+str(sample)+"/boundary.vtk", "w")
    text_file.write('# vtk DataFile Version 2.0\n')
    text_file.write('rawdata, Created by Gmsh\n')
    text_file.write('ASCII\n')
    text_file.write('DATASET POLYDATA\n')
    text_file.write('POINTS '+str(BoundaryList.shape[0])+' double\n')
    for i in range(BoundaryList.shape[0]):
        text_file.write(str(BoundaryList[i,0])+'\t'+str(BoundaryList[i,1])+'\t'+str(0)+'\n')

    text_file.close()



### Generate reference mesh using FreeFem and fetch boundary like before
try:
	os.makedirs('data/reference/')
except:
	print('Sth went wrong, please check')

os.system("FreeFem++ ReferMeshGeneration.edp -v 0")

print('Handling reference mesh')
    
### Open up all necessary documents
TargetMSH = open("data/reference/reference.msh", "r+")

### Read the total number of vertices(points) from .msh file
line = TargetMSH.readline()
words = line.strip().split()
v_num = int(words[0])
t_num = int(words[1])
e_num = int(words[2])

TargetMSH.close()
# print('TOTAL VERTICES:',v_num)

### Create a list and copy out all the boundary vertices
BoundaryList = []
InnerList = []
for i in range(1,v_num+1):
	with open("data/reference/reference.msh",'r') as txt:

		text = txt.readlines()
		currentline = text[i]

		coordinates = currentline.strip().split()
		x = np.round(np.float64(coordinates[0]),decimals=12)
		y = np.round(np.float64(coordinates[1]),decimals=12)
		label =  int(coordinates[2])

		if label != 0:
			BoundaryList.append([x,y,label])
		else:
			InnerList.append([x,y,label]) 

BoundaryList = np.asarray(BoundaryList)
InnerList = np.asarray(InnerList)
# print(BoundaryList.shape)

np.save('data/reference/inner.npy',InnerList)
np.save('data/reference/boundary.npy',BoundaryList)

### Write a VTK files
### Head content: 
###     # vtk DataFile Version 2.0
###     rawdata, Created by Gmsh
###     ASCII
###     DATASET POLYDATA
###     POINTS 2774 double    
text_file = open("data/reference/referboundary.vtk", "w")
text_file.write('# vtk DataFile Version 2.0\n')
text_file.write('rawdata, Created by Gmsh\n')
text_file.write('ASCII\n')
text_file.write('DATASET POLYDATA\n')
text_file.write('POINTS '+str(BoundaryList.shape[0])+' double\n')
for i in range(BoundaryList.shape[0]):
	text_file.write(str(BoundaryList[i,0])+'\t'+str(BoundaryList[i,1])+'\t'+str(0)+'\n')

text_file.close()

print('Step 3 - Boundary vertice fetching: done ')

#################################################################################
### Step 4: Use deformetrica to perform surface registration 
#################################################################################

### Generate script file for deformetrica for 500 samples
### The output folder will be at the same level of directory of 'data'
os.system("python deformetrica_script/writeXML.py")
os.system("time deformetrica estimate deformetrica_script/model.xml \
           deformetrica_script/data_set.xml -p deformetrica_script/optimization_parameters.xml")

print('Step 4 - Surface registration: done ')

#################################################################################
### Step 5: Compute mapping with RBF interpolation 
#################################################################################

### SRmesh is used for validation for the solution got on the reference domain
try:
	os.mkdir('data/mapping/')
	os.mkdir('data/SRmesh/')
except:
	print('Sth went wrong, please check')


### Read in the undeformed data and save the control points
OriBoundary = np.load("data/reference/boundary.npy")
OriInner = np.load("data/reference/inner.npy")
prename = "DeterministicAtlas__Reconstruction__mesh__subject_sample"
np.savetxt('data/mapping/control.txt',np.hstack((OriBoundary[:,0],OriBoundary[:,1])))

matcharray = []
matcharraylabel = []

for sample in range(num_samples):
	if sample%100==0:
		print('Handling sample:',sample)


	### Read in deformed data
	DefBoundary = readVTK("output/"+prename+str(sample)+".vtk")

	### Clean and fix the inlet boundary due to registration
	DefBoundary = cleanBC(DefBoundary,OriBoundary)
	
	### Train RBFI for intepolation, the last column of OriBoundary is the label for the boundary 
	rbf = RBFInterpolator(OriBoundary[:,:2],DefBoundary,kernel='cubic',degree=1)
	DefInner = rbf(OriInner[:,:2])

	# print(rbf._shift)
	# print(rbf._scale)
	# test = manual_rbf(rbf._coeffs,OriBoundary[:,:2],OriInner[:,:2],rbf._shift,rbf._scale)
	# print(np.mean(DefInner-test))

	### Stack together the nodes
	OriNodes = np.vstack((OriBoundary[:,:2],OriInner[:,:2]))
	DefNodes = np.vstack((DefBoundary,DefInner))

	os.mkdir('data/mapping/sample_'+str(sample))
	np.savetxt('data/mapping/sample_'+str(sample)+'/weights.txt',np.hstack((rbf._coeffs[:,0],rbf._coeffs[:,1])))

	### Copy the average.msh to the folder for further modification
	shutil.copyfile('data/reference/reference.msh', 'data/SRmesh/'+str(sample)+'.msh')

	### Total number of vertices
	v_num = OriBoundary.shape[0] + OriInner.shape[0]

	### Open up the .msh file and match it with OriBoundary and OriInner
	with open("data/SRmesh/"+str(sample)+".msh",'r') as txt:
		text = txt.readlines()
		
		if sample == 0:
			for i in range(1,v_num+1):
				currentline = text[i]
				coordinates = currentline.strip().split()
				x = np.round(np.float64(coordinates[0]),decimals=12)
				y = np.round(np.float64(coordinates[1]),decimals=12)
				label =  int(coordinates[2])

				for j in range(v_num):

					if np.sqrt((OriNodes[j,0]-x)**2+(OriNodes[j,1]-y)**2) < 1e-4:
						matcharray.append(j)
						matcharraylabel.append(label)
						break

		for i in range(1,v_num+1):
			### assign value from target.vtk -> source.msh
			newline = str(DefNodes[matcharray[i-1],0]) + ' ' + str(DefNodes[matcharray[i-1],1]) + ' ' + str(matcharraylabel[i-1]) + '\n'
			text[i] = newline
			
		with open("data/SRmesh/"+str(sample)+".msh",'w') as txt:
			txt.writelines(text)

print('Step 5 - Mapping computation: done ')

#################################################################################
### Step 6: Parametrize the geometry and compute reduced parameters of geometry 
#################################################################################

### ROM -> saving data of ROM
### snapshots -> saving direct snapshots data generated from FEM simulation

try:
	os.mkdir('data/ROM/')
	os.mkdir('data/snapshots/')

	for sample in range(num_samples):		
		os.mkdir('data/snapshots/sample_'+str(sample))

except:
	print('Sth went wrong, please check')



### Extract boundary from vtk files
for sample in range(num_samples):
	if sample%100 == 0:
		print('Handling sample:',sample)
	
	TargetMesh = open("data/SRmesh/"+str(sample)+".msh", "r")

	### Read basic information of mesh
	line = TargetMesh.readline()
	words = line.strip().split()
	v_num = int(words[0])
	t_num = int(words[1])
	e_num = int(words[2])

	TargetMesh.close()
	
	### Create a list and copy out all the boundary vertices
	BoundaryList = []

	with open("data/SRmesh/"+str(sample)+".msh",'r') as txt:
		text = txt.readlines()
		
		for i in range(1,v_num+1):	
			currentline = text[i]

			coordinates = currentline.strip().split()
			x = np.round(np.float64(coordinates[0]),decimals=12)
			y = np.round(np.float64(coordinates[1]),decimals=12)
			label = int(coordinates[2])

			if label !=0:
				BoundaryList.append([x,y,label])
	
	np.save('data/raw/sample_'+str(sample)+'/SRboundary.npy',np.asarray(BoundaryList))



RefList = np.load('data/reference/boundary.npy')
# print(RefList.shape)

GeoPara = np.empty((0,RefList.shape[0]*2))
# print(GeoPara.shape)

for i in range(num_samples):
	GeoList = np.load('data/raw/sample_'+str(i)+'/SRboundary.npy')

	xcoord = GeoList[:,0] - RefList[:,0]
	ycoord = GeoList[:,1] - RefList[:,1]
	coord = np.hstack((xcoord,ycoord))

	GeoPara = np.vstack((GeoPara,np.expand_dims(coord,axis=0)))

Mean = np.expand_dims(np.mean(GeoPara,axis=0),axis=0)
GeoPara_hat = GeoPara - Mean

### Define setting for PCA
NumComponent=17
# pca = PCA(n_components=NumComponent, svd_solver='full')
pca = PCA(n_components=NumComponent)
pca.fit(GeoPara_hat)

print('PCA process finished, and here is the properties of PCA')
print('Explained_variance_ratio:',np.sum(pca.explained_variance_ratio_))
print('Singular_valuesl:',pca.singular_values_)
print('Components:',pca.components_.shape)
print('N_features:',pca.n_features_)
print('N_samples:',pca.n_samples_)

### Generate projection coefficient alpha(t,mu) matrix = (10*2601)*(2601*12000) = 10*12000
Alpha = pca.transform(GeoPara_hat)
print(Alpha.shape)

np.save('data/ROM/Reduced_GeoPara.npy',Alpha)

print('Step 6 - Reduced geometric paramterization: done ')