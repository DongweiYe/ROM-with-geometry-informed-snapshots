import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.interpolate import RBFInterpolator


def extract_vertices(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    l_num = int(np.ceil(v_num/5))
    TargetMSH.close()
    # print('TOTAL VERTICES:',v_num)

    ### Create a list and copy out all the boundary vertices
    VerticeList = []

    for i in range(1,l_num+1):
        with open(directory_to_txt,'r') as txt:

            text = txt.readlines()
            currentline = text[i]
            coordinates = currentline.strip().split()

            for j in range(len(coordinates)):
                VerticeList.append(np.float64(coordinates[j]))
    return np.expand_dims(np.asarray(VerticeList),axis=1)


def vertices_num(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    TargetMSH.close()
    return v_num


def check_num_snapshot(Attri,directory_to_data,num_snapshot,num_rep):
    SnapshotMatrix = np.load(directory_to_data+"/ROM/POD"+Attri+".npy")
    NumSample = SnapshotMatrix.shape[1]
    TrainRatio = 0.8
    NumTrain = int(NumSample*TrainRatio)

    ### Split out one case for test
    TrainCase = SnapshotMatrix[:,:NumTrain]
    TestCase = SnapshotMatrix[:,NumTrain:]

    ### Further choose the small trainnig dataset and validation dataset
    ValidationCase = TestCase[:,:50]
    error_array = np.zeros(num_rep)
    for loop in range(num_rep):
        idx = np.random.randint(400, size=num_snapshot)
        SmallTrainCase = TrainCase[:,idx]

        ### Calculate mean
        Mean = np.expand_dims(np.mean(SmallTrainCase,axis=1),axis=1)
        SmallTrainCase_hat = SmallTrainCase - Mean
        ValidationCase_hat = ValidationCase - Mean

        ratio = 0
        num_component = 5
        while ratio<0.999:
            pca = PCA(n_components=num_component, svd_solver='full')
            pca.fit(SmallTrainCase_hat.T)

            # print('Testing variance_ratio:',np.sum(pca.explained_variance_ratio_))

            ratio=np.sum(pca.explained_variance_ratio_)
            if ratio>0.999:
                # print('PCA process finished, and here is the properties of PCA')
                # print('Explained_variance_ratio:',np.sum(pca.explained_variance_ratio_))
                # print('Singular_valuesl:',pca.singular_values_)
                # print('Components:',pca.components_.shape)
                # print('N_features:',pca.n_features_)
                # print('N_samples:',pca.n_samples_)
                pass
            else:
                num_component = num_component + 1
        
        AfterProjection = pca.inverse_transform(pca.transform(ValidationCase_hat.T)).T + Mean
        # print('L2 norm error:',np.mean(compute_l2_error(AfterProjection.T,ValidationCase_hat)))
        error_array[loop] = np.mean(compute_l2_error(AfterProjection,ValidationCase))

    # print(error_array)
    return np.mean(error_array),np.std(error_array)


def SVD(Attri,NumComponent,TrainRatio,directory_to_data,save_data):
    SnapshotMatrix = np.load(directory_to_data+"/ROM/POD"+Attri+".npy")
    NumSample = SnapshotMatrix.shape[1]
    TrainRatio = 0.8
    NumTrain = int(NumSample*TrainRatio)

    ### Split out one case for test
    TrainCase = SnapshotMatrix[:,:NumTrain]
    TestCase = SnapshotMatrix[:,NumTrain:]

    ### Calculate mean
    Mean = np.expand_dims(np.mean(TrainCase,axis=1),axis=1)
    TrainCase_hat = TrainCase - Mean
    TestCase_hat = TestCase - Mean

    pca = PCA(n_components=NumComponent, svd_solver='full')
    pca.fit(TrainCase_hat.T)

    print('PCA process finished, and here is the properties of PCA')
    print('Explained_variance_ratio:',np.sum(pca.explained_variance_ratio_))
    print('Singular_valuesl:',pca.singular_values_)
    print('Components:',pca.components_.shape)
    print('N_features:',pca.n_features_)
    print('N_samples:',pca.n_samples_)

    ### Generate projection coefficient alpha(mu) matrix = (10*260)*(2601*12000) = 10*12000
    Alpha = pca.transform(TrainCase_hat.T)
    AlphaTest = pca.transform(TestCase_hat.T)

    ### Check the error of POD based on validation dataset and test dataset
    ReconstTest = pca.inverse_transform(AlphaTest).T + Mean
    ValidError = compute_l2_error(ReconstTest[:,:50],TestCase[:,:50])
    TestError = compute_l2_error(ReconstTest[:,50:],TestCase[:,50:])
    print('Validation error: ',np.mean(ValidError),'%; Test error: ',np.mean(TestError),'%')
    print('Validation sd: ',np.std(ValidError),'%; Test sd: ',np.std(TestError),'%')


    ### Transform the data(output) to NN training type 
    ReducedCoef = np.vstack((Alpha,AlphaTest))
    print('Training output of '+Attri+' shape: ',ReducedCoef.shape,'\n')

    if save_data == True:
        np.save(directory_to_data+'/ROM/ReducedFlow_'+Attri+'.npy',ReducedCoef)
        np.save(directory_to_data+'/ROM/Components_'+Attri+'.npy',pca.components_)
        np.save(directory_to_data+'/ROM/Mean_'+Attri+'.npy',Mean)
    

def RBFI_function(Attri,directory_to_data,save_data):
    ### Load the data input&output
    InputCP = np.load(directory_to_data+'/ROM/Reduced_GeoPara.npy')
    Output = np.load(directory_to_data+'/ROM/ReducedFlow_'+Attri+'.npy')

    print('Size of input CP:', InputCP.shape)
    print('Size of output '+Attri+' :',Output.shape)

    ### scale data to 0-1 (normalization)
    scalerCP = preprocessing.MinMaxScaler()
    NormalCP = scalerCP.fit_transform(InputCP)

    ### normlize output (optional)
    scalerOut = preprocessing.MinMaxScaler()
    NormalOutput = scalerOut.fit_transform(Output)

    ### RBF interpolation
    rbf_function = RBFInterpolator(NormalCP[:400,:], NormalOutput[:400,:],kernel='cubic',degree=1)
    
    start = time.time()
    prediction = rbf_function(NormalCP[400:,:])

    ### Reverse the normalization
    prediction = scalerOut.inverse_transform(prediction)
    end = time.time()
    print('Average RBFI computational time for '+Attri+' is:',(end - start)/100)

    ### Plot check
    # number = 100
    # for i in range(18):
    #     plt.plot(Output[400:(400+number),i:i+1],'*',label='trut')
    #     plt.plot(prediction[0:number,i:i+1],'*',label='pred')
    #     plt.legend()
    #     plt.savefig(Attri+'_'+str(i)+'.png')
    #     plt.clf()

    ### save data
    if save_data == True:
        np.save('data/ROM/pred_reduced_coef_'+Attri+'.npy',prediction)


def coef_to_field(Attri,directory_to_data):

    ### Load prediction, pca components, and means
    alpha = np.load(directory_to_data+'/ROM/pred_reduced_coef_'+Attri+'.npy')
    pca_components = np.load(directory_to_data+'/ROM/Components_'+Attri+'.npy')
    mean = np.load(directory_to_data+'/ROM/Mean_'+Attri+'.npy')
    groundtruth = np.load(directory_to_data+"/ROM/POD"+Attri+".npy")

    ### Set test/validation cases
    StartNumTest = 400
    NumTest = 50
    TestCase = groundtruth[:,StartNumTest:(StartNumTest+NumTest)]

    ### Recover the variable field
    start = time.time()
    Prediction_wo_mean = alpha@pca_components
    Prediction = Prediction_wo_mean.T + mean
    end = time.time()
    print('Average resemble computational time for '+Attri+' is:',(end - start)/100)

    ### Compute test/validation error
    print('Mean error:',np.mean(compute_l2_error(Prediction[:,(StartNumTest-400):(StartNumTest-400+NumTest)],TestCase)))
    print('STD error:',np.std(compute_l2_error(Prediction[:,(StartNumTest-400):(StartNumTest-400+NumTest)],TestCase)))

    ### format the data into FreeFem txt form
    vertice = Prediction.shape[0]
    try:
        os.mkdir(directory_to_data+'/error/')
    except:
        print('Sth went wrong, please check')

    for i in range(400,500):
    	# print('Processing sample ',str(i),':')
    	### Create a error directory
        try:
            os.mkdir(directory_to_data+'/error/sample_'+str(i))
        except:
            pass
    	## Save a Freefem readable file
        AllString = str(vertice)+'\t\n'
        for k in range(1,vertice+1):
            if k%5==0:
                AllString = AllString+"\t"+str(Prediction[k-1,i-400])+"\n"
            else:
                AllString = AllString+"\t"+str(Prediction[k-1,i-400])
        
        AllString = AllString+"\t"
        text_file = open(directory_to_data+"/error/sample_"+str(i)+'/'+Attri+'pred.txt', "w")
        text_file.write(AllString)
        text_file.close()
    print('Transformation to Freefem txt for ',Attri,' is done')


def compute_l2_error(target,reference):
	return np.sqrt(np.sum(np.square(target-reference),axis=0)/np.sum(np.square(target),axis=0))*100

###############################################
##############   Main ROM process #############
###############################################

### Collecting the data generated from FreeFem
### Snapshot data directory

raw_data_directory = 'data/snapshots'

### Read number of vertice and create emtpy matrix for data storage
v_num = vertices_num(raw_data_directory+'/sample_0/u.txt')
# print(v_num)
ulist = np.empty((v_num,0))
vlist = np.empty((v_num,0))
plist = np.empty((v_num,0))

### Collect all the snapshots  
for sample in range(500):
	print('Processing sample:',sample)
	u = extract_vertices(raw_data_directory+'/sample_'+str(sample)+'/u.txt')
	v = extract_vertices(raw_data_directory+'/sample_'+str(sample)+'/v.txt')
	p = extract_vertices(raw_data_directory+'/sample_'+str(sample)+'/p.txt')

	ulist = np.hstack((ulist,u))
	vlist = np.hstack((vlist,v))
	plist = np.hstack((plist,p))

### Save snapshot matrix (v_num,num_sample)
np.save('data/ROM/PODu.npy',ulist)
np.save('data/ROM/PODv.npy',vlist)
np.save('data/ROM/PODp.npy',plist)
print('Snapshot matrix shape: ', ulist.shape)


## Compute the error of decompsition of different size of snapshot matrix
# for i in range(50,400,50):
#     print('Snapshot: ',i)
#     mean,std = check_num_snapshot('v','data',i,20)
#     print(mean,std)


### Apply PCA to extract the reduced basis and reduced coefficients.
### Define setting for  steady, u->27, v->28, p->19
SVD('u',27,0.8,'data',save_data=True)
SVD('v',28,0.8,'data',save_data=True)
SVD('p',19,0.8,'data',save_data=True)

### RBFI interpolation
RBFI_function('u','data',save_data=True)
RBFI_function('v','data',save_data=True)
RBFI_function('p','data',save_data=True)

### Recover the reduced coefficient back to the variable fields
coef_to_field('u','data')
coef_to_field('v','data')
coef_to_field('p','data')

## Execute Freefem to create a complete vtk file for paraview visualization
os.system("FreeFem++ Visualization.edp -v 0")



