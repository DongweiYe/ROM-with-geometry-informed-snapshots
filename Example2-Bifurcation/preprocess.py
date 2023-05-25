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


def extract_vertices(directory_to_msh):
    TargetMSH = open(directory_to_msh, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])

    TargetMSH.close()
    print('TOTAL VERTICES:',v_num)

    ### Create a list and copy out all the boundary vertices
    BoundaryList = []
    InnerList = []

    for i in range(1,v_num+1):
        with open(directory_to_msh,'r') as txt:

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

    return BoundaryList, InnerList

### Save the points as point cloud in the vtk files
def savevtk(directory_to_vtk,BList):
    ### Write a VTK files
    ### Head content: 
    ###     # vtk DataFile Version 2.0
    ###     rawdata, Created by Gmsh
    ###     ASCII
    ###     DATASET POLYDATA
    ###     POINTS 2774 double    
    text_file = open(directory_to_vtk, "w")
    text_file.write('# vtk DataFile Version 2.0\n')
    text_file.write('rawdata, Created by Gmsh\n')
    text_file.write('ASCII\n')
    text_file.write('DATASET POLYDATA\n')
    text_file.write('POINTS '+str(BList.shape[0])+' double\n')
    for i in range(BList.shape[0]):
        text_file.write(str(BList[i,0])+'\t'+str(BList[i,1])+'\t'+str(0)+'\n')
    text_file.close()

########################################################################################
###   Note the procedure of example 2 is slightly different from example 1
###   This is a data generation and preprocessing files including following process 
###         1) Refernece mesh generation 
### 		2) Deform reference meshes with samples on the displacement of control points
### 		* In this case, the boundary information of each geometry is achieved directly.
###           Unlike in example 1, we generate meshes for each geometry first then fetch the 
###			  boundary information.
###         3) Surface registration (using deformetrica)
###         4) Compute the mapping using RBFI and save the data for FEM simulation
###			5) Parametrize the geometry and compute reduced parameters of geometry
########################################################################################


### Generate reference mesh using FreeFem
try:
	os.makedirs('data/reference/')
except:
	print('Sth went wrong, please check')

os.system("FreeFem++ ReferMeshGeneration.edp -v 0")

### Extract boundary and inner point of a reference domain and save
RefBoundary, RefInner = extract_vertices('data/reference/reference.msh')
RefVertice = np.vstack((RefBoundary,RefInner))
savevtk('data/reference/referboundary.vtk',RefBoundary)
np.save('data/reference/boundary.npy',RefBoundary)
np.save('data/reference/inner.npy',RefInner)

print('Step 1 - Refernece mesh generation: done ')

### Process fixing CP points 
OutCP1 = RefBoundary[np.where(RefBoundary[:,2]==3)[0],:2]
OutCP2 = RefBoundary[np.where(RefBoundary[:,2]==6)[0],:2]
OutCP3 = RefBoundary[np.where(RefBoundary[:,2]==9)[0],:2]
OutCP4 = RefBoundary[(RefBoundary[:,0]>2)&(RefBoundary[:,0]<2.5)][:,:2]
OutCP = np.vstack((OutCP1,OutCP2,OutCP3,OutCP4))

### Process deforming CP points 
BraCP1 = RefBoundary[np.where(RefBoundary[:,2]==7)[0],:2][60,:2]
BraCP2 = RefBoundary[np.where(RefBoundary[:,2]==5)[0],:2][60,:2]
BraCP3 = RefBoundary[np.where(RefBoundary[:,2]==4)[0],:2][60,:2]
BraCP4 = RefBoundary[np.where(RefBoundary[:,2]==2)[0],:2][60,:2]

DefCP = np.asarray([[10,0],
                    [10,2]])
DefCP = np.vstack((DefCP,BraCP1,BraCP2))


num_samples = 500
A = lhs(8, samples=num_samples)
A = A-0.5

### Apply displacement 
DefCPflate = np.expand_dims(np.hstack((DefCP[:,0],DefCP[:,1])),axis=0)
DisCP = A + DefCPflate

### Create data folder
try:
	os.makedirs('data/raw/')
except:
	print('Sth went wrong, please check')

for i in range(num_samples):
    DisCPx = np.expand_dims(DisCP[i,:4],axis=1)
    DisCPy = np.expand_dims(DisCP[i,4:],axis=1)

    TrainInput = np.vstack((DefCP,OutCP))
    TrainOutput = np.vstack((np.hstack((DisCPx,DisCPy)),OutCP))
    rbf = RBFInterpolator(TrainInput,TrainOutput,kernel='cubic')
    DefVertice = rbf(RefBoundary[:,:2])
    os.mkdir('data/raw/sample_'+str(i))	
    savevtk('data/raw/sample_'+str(i)+'/boundary.vtk',DefVertice)

print('Step 2 - Geometry generation of 500 samples : done ')


#################################################################################
### Step 3: Use deformetrica to perform surface registration 
#################################################################################

### Generate script file for deformetrica for 500 samples
### The output folder will be at the same level of directory of 'data'
os.system("python deformetrica_script/writeXML.py")
os.system("time deformetrica estimate deformetrica_script/model.xml \
           deformetrica_script/data_set.xml -p deformetrica_script/optimization_parameters.xml")

print('Step 4 - Surface registration: done ')

#################################################################################
### Step 4: Compute mapping with RBF interpolation 
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
### Step 5: Parametrize the geometry and compute reduced parameters of geometry 
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
NumComponent=8
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