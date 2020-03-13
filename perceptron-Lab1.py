import numpy as np
import matplotlib.pyplot as plt
import random

n = 100
mA = [1, 0.3]
sigmaA = 0.1
mB = [0.2, 3.8]
sigmaB = 0.1
#classA=np.concatenate([np.random.normal(mean, sigmaA, (1,n)) for mean in mA], axis = 0)
#classA=np.concatenate([np.random.normal(mean, sigmaA, (1,n)) for mean in mA], axis = 0
classA=np.random.normal(0,sigmaA,(1,int(0.5*n)))-mA[0]*np.ones((1,int(0.5*n)))
classA=np.concatenate([classA,np.random.normal(0,sigmaA,(1,int(0.5*n)))+mA[0]*np.ones((1,int(0.5*n)))],axis=1)
classA=np.concatenate([classA,np.random.normal(mA[1], sigmaA, (1,n))], axis = 0)
classA=np.concatenate((classA,np.ones((1,n))),axis=0)
classB=np.concatenate([np.random.normal(mean, sigmaB, (1,n)) for mean in mB], axis = 0)
classB=np.concatenate((classB,-1*np.ones((1,n))),axis=0)
data = np.concatenate([classA, classB], axis = 1)
data = data[:,np.random.permutation(range(data.shape[1]))]

grid=np.array([[(i,j) for i in range(-50,50)] for j in range(-50,50)],dtype=np.float32)

grid = grid*0.1

autoencoder_data=[-1*np.ones((1,8)) for i in range(n)]

X = np.vstack([data[0:2, :], np.ones(data.shape[1])])
label=data[2,:].reshape((1, data.shape[1]))
#label=data[2,:]
label1=data[2]

def Plotting(classA, classB,W):
    plt.plot(classA[0], classA[1], marker='o', color='red', ls='')
    plt.plot(classB[0], classB[1], marker='x', color='blue', ls='')
    plt.plot([-5, 0.5], [-W[0] * (-5) / W[1] - W[2] / W[1], -W[0] * (0.5) / W[1] - W[2] / W[1]], color='green')
    plt.xlabel('Data x value')
    plt.ylabel('Data y value')
    plt.show()


def delta_calc(W,ro,x,y,sol):
    if W.shape[1]>2:
        error=sol-W[0][0]*x-W[0][1]*y-W[0][2]
        deltaW = np.array([ro * error * x, ro * error * y, ro * error])
    else:
        error=sol-W[0][0]*x-W[0][1]*y
        deltaW=np.array([ro*error*x,ro*error*y])
    return W+deltaW

def perceptron_calc(ro,out,label,features):
    error=label-out
    print(features)
    return ro*np.matmul(features,np.transpose(error))

def perceptron_batch(features):
    mW=0
    sigmaW=0.4
    W0 = np.random.normal(mW, sigmaW, (1, features.shape[0]))
    Wdelta=W0
    label=features[2]
    features=np.delete(features,2,0)
    print(features.shape[1])
    features=np.concatenate([features,np.ones((1,features.shape[1]))],axis=0)
    ro=0.1
    epoch=50
    for e in range(epoch):
        X=np.matmul(W0[0],features)
        for j in range(X.shape[0]):
            if X[j]>0:
                X[j]=1
            else:
                X[j]=-1

        Winc=perceptron_calc(ro,X,label,features)
        Wdelta=[Wdelta[0]-Winc.T]

    return Wdelta

def perceptron1a1_NO_Bias(features):
    mW = 0
    sigmaW = 0.4
    W0 = np.random.normal(mW, sigmaW, (1, features.shape[0]-1)) #Por que me salen dobles corchetes: porque le he especificado el tamaño en dos partes
    Wdelta=W0                                      # si en size hubiera sido solo features.shape[0], se hubiera escrito en un unico vector

    ro=0.15
    epoch=50
    for e in range(epoch):
        for i in range(0,features.shape[1]):
            out=Wdelta[0][0]*features[0][i]+Wdelta[0][1]*features[1][i]
            if out>0:
                out=1
            else:
                out=-1
            if out!=data[2][i]:
                Winc=perceptron_calc(ro,np.array([out]),np.array([features[2][i]]),np.array([features[0][i],features[1][i]]).reshape(2,1))
                Wdelta=np.array([Wdelta[0]+Winc])
    return Wdelta

def perceptron1a1(features):
    mW = 0
    sigmaW = 0.4
    W0 = np.random.normal(mW, sigmaW, (1, features.shape[0])) #Por que me salen dobles corchetes: porque le he especificado el tamaño en dos partes
    Wdelta=W0                                      # si en size hubiera sido solo features.shape[0], se hubiera escrito en un unico vector

    ro=0.15
    epoch=50
    for e in range(epoch):
        for i in range(0,features.shape[1]):
            out=Wdelta[0][0]*features[0][i]+Wdelta[0][1]*features[1][i]+Wdelta[0][2]
            if out>0:
                out=1
            else:
                out=-1
            if out!=data[2][i]:
                Winc=perceptron_calc(ro,np.array([out]),np.array([features[2][i]]),np.array([features[0][i],features[1][i],1]).reshape(3,1))
                Wdelta=np.array([Wdelta[0]+Winc])
    return Wdelta

def delta_learning1a1_NO_Bias(features):
    mW=0
    sigmaW=0.4
    label = features[2]
    features = np.delete(features, 2, 0)
    W0=np.random.normal(mW, sigmaW, (1,features.shape[0]))
    print(W0[0])
    Wdelta=W0
    ro=0.15
    epoch=50
    for e in range(epoch):
        for i in range(0,features.shape[1]):
            out=Wdelta[0][0]*features[0][i]+Wdelta[0][1]*features[1][i]
            if out>0:
                classif=1
            else:
                classif=-1
            if classif!=label[i]:
                Wdelta=delta_calc(Wdelta,ro,features[0][i],features[1][i],label[i])
    return Wdelta

def delta_learning1a1(features):
    mW=0
    sigmaW=0.4
    W0=np.random.normal(mW, sigmaW, (1,features.shape[0]))
    print(W0[0])
    Wdelta=W0
    ro=0.15
    epoch=50
    for e in range(epoch):
        for i in range(0,features.shape[1]):
            out=Wdelta[0][0]*features[0][i]+Wdelta[0][1]*features[1][i]+Wdelta[0][2]
            if out>0:
                classif=1
            else:
                classif=-1
            if classif!=data[2][i]:
                Wdelta=delta_calc(Wdelta,ro,features[0][i],features[1][i],features[2][i])
    return Wdelta


def error_Batch(W,X,answer,eta):
    return eta*(np.matmul(np.matmul(W,X)-answer,X.T)),(np.matmul(W,X)-answer)


def delta_learning_Batch(X,label,eta):
    mW = 0
    sigmaW = 0.4
    W0 = np.random.normal(mW, sigmaW, ((label.shape[0], X.shape[0])))
    print(W0[0])
    W = W0[0]
    epoch=50
    errorList = []
    for i in range(epoch):
        deltaW,error=error_Batch(W,X,label,eta)
        W=W-deltaW
        errorList.append(np.mean(error))

    return W,errorList

def delta_learning_Batch_NO_Bias(X,label,eta):
    mW = 0
    sigmaW = 0.4
    W0 = np.random.normal(mW, sigmaW, ((label.shape[0], X.shape[0])))
    print(W0[0])
    W = W0[0]
    epoch=50
    errorList = []
    for i in range(epoch):
        deltaW,error=error_Batch(W,X,label,eta)
        W=W-deltaW
        errorList.append(np.mean(error))

    return W,errorList

def two_layer_Delta_Batch_error(out_hidden,out_net,act_net,W,label):
    deriv1=sigmoid_derv(out_net)
    deltah=np.multiply((label-act_net),deriv1)

    deriv2=sigmoid_derv(out_hidden)
    deltak=np.multiply(np.matmul(W.T,deltah),deriv2)
    deltak=deltak[:out_hidden.shape[0]-1,:]
    return deltah,deltak

def two_layer_verification(val,V,W):
    count=0
    label = val[2].reshape((1, val.shape[1]))
    val = np.delete(val, 2, 0)
    output_hidden = np.matmul(V, val)
    act_hidden = output_hidden
    for j in range(V.shape[0]):
        for k in range(output_hidden.shape[1]):
            if output_hidden[j][k] > 0:
                act_hidden[j][k] = 1
            else:
                act_hidden[j][k] = -1
    output_net = np.matmul(W, act_hidden)
    act_net = output_net
    for j in range(output_net.shape[0]):
        for k in range(output_net.shape[1]):
            if output_net[j][k] > 0:
                act_net[j][k] = 1
            else:
                act_net[j][k] = -1
    for i in range(label.shape[1]):
        if act_net[0][i]!=label[0][i]:
            count=count+1
    return count

def sigmoid_derv(data):
    return np.multiply(data,1-data)

def sigmoid(data):
    return 2/(1+np.exp(-data))-1

def step(data):
    for i in range(data.shape[0]):
        if data[i]>0:
            data[i]=1
        else:
            data[i]=-1
    return data

def step_derv(out_net):
    return np.multiply(np.ones((1, out_net.shape[1])) + out_net, np.ones((1, out_net.shape[1])) + out_net) / 2

def two_layer_verification_Bias(val,V,W):
    count=0
    label = val[2].reshape((1, val.shape[1]))
    val = np.delete(val, 2, 0)
    val=np.concatenate([val,np.ones((1,val.shape[1]))],axis=0)
    output_hidden = np.matmul(V, val)
    act_hidden = output_hidden[:]
    for j in range(V.shape[0]):
        #act_hidden[j]=step(output_hidden[j])
        act_hidden[j] = sigmoid(output_hidden[j])
    act_hidden=np.concatenate((act_hidden,np.ones((1,val.shape[1]))),axis=0)
    output_net = np.matmul(W, act_hidden)
    act_net = output_net[:]
    for j in range(output_net.shape[0]):
        act_net[j]=step(output_net[j])
        #act_net[j] = sigmoid(output_net[j])
    for i in range(label.shape[1]):

        if act_net[0][i]!=label[0][i]:
            count=count+1
    return count

def nonlinear_delta_No_Bias(data,size_hidden_layer,size_output,val):
    mW=0
    mV=0
    sigma=0.4
    V0=np.random.normal(mV, sigma, ((size_hidden_layer,2)))
    V=V0
    W0=np.random.normal(mW, sigma, ((size_output,size_hidden_layer)))
    W=W0
    ro=0.001
    alfa=0.9
    label=data[2].reshape((1,data.shape[1]))
    data=np.delete(data,2,0)
    epoch=40
    errorList=[]
    error_train=[]
    for e in range(epoch):
        output_hidden=np.matmul(V,data)
        act_hidden=output_hidden
        for i in range(size_hidden_layer):
            for j in range(output_hidden.shape[1]):
                if output_hidden[i][j]>0:
                    act_hidden[i][j]=1
                else:
                    act_hidden[i][j]=-1
        output_net=np.matmul(W,act_hidden)
        act_net=output_net
        for i in range(output_net.shape[0]):
            for j in range(output_net.shape[1]):
                if output_net[i][j]>0:
                    act_net[i][j]=1
                else:
                    act_net[i][j]=-1
        errorW,errorV=two_layer_Delta_Batch_error(output_hidden,output_net,act_net,W,label)
        deltaw=ro*np.matmul(errorW,act_hidden.T)
        deltav=ro*np.matmul(errorV,data.T)
        W=W*alfa-(1-alfa)*deltaw
        V=V* alfa - (1 - alfa) * deltav
        error=two_layer_verification(val,V,W)
        errorList.append(error/val.shape[1])
        error_train.append(two_layer_verification(np.concatenate((data,label),axis=0),V,W))
    return V,W,errorList,error_train

def nonlinear_delta_Bias(data,size_hidden_layer,size_output,val):
    mW=0.2
    mV=-0.5
    sigma=0.5
    V0=np.random.normal(mV, sigma, ((size_hidden_layer,2+1)))
    V=V0
    W0=np.random.normal(mW, sigma, ((size_output,size_hidden_layer+1)))
    W=W0
    ro=0.0001
    alfa=0.9
    label=data[2].reshape((1,data.shape[1]))
    data=np.delete(data,2,0)
    data = np.concatenate([data, np.ones((1,data.shape[1]))], axis=0)
    epoch=20
    errorList=[]
    error_train=[]
    for e in range(epoch):
        output_hidden=np.matmul(V,data)
        act_hidden=np.copy(output_hidden)
        for i in range(size_hidden_layer):
            #act_hidden[i]=step(output_hidden[i])
            act_hidden[i] = sigmoid(output_hidden[i])

        act_hidden=np.concatenate([act_hidden, np.ones((1, act_hidden.shape[1]))], axis=0)
        output_hidden = np.concatenate([output_hidden, np.ones((1, output_hidden.shape[1]))], axis=0)
        output_net=np.matmul(W,act_hidden)
        act_net=np.copy(output_net)
        for i in range(output_net.shape[0]):
            #act_net[i]=step(output_net[i])
            act_net[i] = sigmoid(output_net[i])
        errorW,errorV=two_layer_Delta_Batch_error(output_hidden,output_net,act_net,W,label)
        deltaw=-ro*np.matmul(errorW,act_hidden.T)
        deltav=-ro*np.matmul(errorV,data.T)
        #W=W*alfa+(1-alfa)*deltaw
        #V=V* alfa + (1 - alfa) * deltav
        W=W+deltaw
        V=V+deltav
        #a=np.concatenate((val, np.ones((1, val.shape[1]))), axis=0)
        error=two_layer_verification_Bias(val,V,W)
        errorList.append(error/val.shape[1])
        data_test=np.concatenate((data[0:2],label),axis=0)
        data_test=np.concatenate((data_test,np.ones((1,data.shape[1]))),axis=0)
        error_train.append(two_layer_verification_Bias(data_test[:data_test.shape[0]-1],V,W))
        #if (error/val.shape[1])<0.05 or (e>1 and errorList[e]>errorList[e-1]):
        #    break
    return V,W,errorList,error_train

def nonlinear_delta_Bias_Seq(data,size_hidden_layer,size_output,val):
    mW = 0
    mV = -0.1
    sigma = 0.4
    V0 = np.random.normal(mV, sigma, ((size_hidden_layer, 2 + 1)))
    V = V0
    W0 = np.random.normal(mW, sigma, ((size_output, size_hidden_layer + 1)))
    W = W0
    ro = 0.003
    label = data[2].reshape((1, data.shape[1]))
    data = np.delete(data, 2, 0)
    data = np.concatenate([data, np.ones((1, data.shape[1]))], axis=0)
    epoch = 40
    alpha=0.9
    errorList = []
    error_epoch = []
    first_layer_output=np.zeros((size_hidden_layer,data.shape[0]))
    for e in range(epoch):
        print(e)
        print(errorList)
        for i in range(data.shape[1]):
            #First we do the forwards pass:
            first_layer_output=np.matmul(V,np.array([data[0][i], data[1][i], data[2][i]]).reshape(3,1))
            act_first_layer=sigmoid(first_layer_output)
            act_first_layer=np.concatenate([act_first_layer,np.ones((1,1))],axis=0)
            #Second layer
            second_layer_output=np.matmul(W,act_first_layer)
            act_second_layer=sigmoid(second_layer_output)

            #Now, we will calculate the back propagation
            deriv1=sigmoid_derv(second_layer_output)
            error1=(act_second_layer-label[0][i])*deriv1[0]
            deriv2=sigmoid_derv(np.concatenate([first_layer_output,np.ones((1,1))],axis=0)) #Aqui han quitado un punto del vector que es lo que lo jodeee
            a=np.matmul(W.T,error1)
            error2=np.multiply(np.matmul(W.T,error1),deriv2)
            error2=error2[:size_hidden_layer,:]
            dw=-ro*error1*act_first_layer.T
            dv=-ro*error2*np.array([data[0][i], data[1][i], data[2][i]]).reshape(1,3)
            dw=dw*alpha-error1*act_second_layer*(1-alpha)
            dv=dv*alpha-error2*act_first_layer[:size_hidden_layer,:]*(1-alpha)
            W=W+dw*ro
            V=V+dv*ro
            error=two_layer_verification_Bias(val,V,W)
            errorList=np.append(errorList,error)
        error_epoch=np.append(error_epoch,np.mean(errorList))
        error_train=two_layer_verification_Bias(data,V,W)
        print(error_train/data.shape[1])
        if error_epoch.shape[0]>1:
            if error_epoch[e]<=error_epoch[e-1]:
                Vbest=V
                Wbest=W
            else:
                break
        else:
            Vbest=V
            Wbest=W
        #if error_epoch[e]<0.05*val.shape[1]:
        #    break
    errorList=errorList/val.shape[1]
    return Vbest,Wbest,error_epoch

def autoencoder(data,size_hidden_layer,size_output,val):
    mW = 0
    mV = 0.5
    sigma = 0.4
    V0 = np.random.normal(mV, sigma, ((size_hidden_layer, 8 + 1)))
    V = V0
    W0 = np.random.normal(mW, sigma, ((size_output, size_hidden_layer + 1)))
    W = W0
    ro = 0.03
    label = data[:]
    for i in range(data.__len__()):
        data[i]=np.append(data[i],1).reshape(1,-1)
    epoch = 40
    alpha=0.9
    errorList = []
    error_epoch = []
    first_layer_output=np.zeros((size_hidden_layer,data[0].shape[1]))
    for e in range(epoch):
        print(e)
        print(errorList)
        for i in range(data.__len__()):
            #First we do the forwards pass:
            first_layer_output=np.matmul(V,data[i].reshape(9,1))
            act_first_layer=sigmoid(first_layer_output)
            act_first_layer=np.concatenate([act_first_layer,np.ones((1,1))],axis=0)
            #Second layer
            second_layer_output=np.matmul(W,act_first_layer)
            act_second_layer=sigmoid(second_layer_output)
            #if act_second_layer>0:
             #   act_second_layer=1
            #else:
             #   act_second_layer=-1

            #Now, we will calculate the back propagation
            deriv1=sigmoid_derv(second_layer_output)
            error1=(act_second_layer-label[i].T)*deriv1
            deriv2=sigmoid_derv(np.concatenate([first_layer_output,np.ones((1,1))],axis=0)) #Aqui han quitado un punto del vector que es lo que lo jodeee
            a=np.matmul(W.T,error1)
            error2=np.multiply(np.matmul(W.T,error1),deriv2)
            error2=error2[:size_hidden_layer,:]
            dw=-ro*error1*act_first_layer.T
            dv=-ro*error2*np.array([data[i]]).reshape(1,-1)
            dw=dw*alpha-error1*act_second_layer*(1-alpha)
            dv=dv*alpha-error2*act_first_layer[:size_hidden_layer,:]*(1-alpha)
            W=W+dw*ro
            V=V+dv*ro
            error=two_layer_verification_Bias(val,V,W)
            errorList=np.append(errorList,error)
        error_epoch=np.append(error_epoch,np.mean(errorList))
        error_train=two_layer_verification_Bias(data,V,W)
        print(error_train/data.shape[1])
        if error_epoch.shape[0]>1:
            if error_epoch[e]<=error_epoch[e-1]:
                Vbest=V
                Wbest=W
            else:
                break
        else:
            Vbest=V
            Wbest=W
        if error_epoch[e]<0.05*val.shape[1]:
            break
    errorList=errorList/val.shape[1]
    return Vbest,Wbest,error_epoch



def errorPlot(errorList):
    plt.plot(np.arange(errorList.__len__()), errorList, marker='o', color='red')
    plt.show()

def Plotting_2_layer(classA,classB,V,W,grid):
    grid_info=np.zeros((grid.shape[0],grid.shape[1]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            data=np.array([grid[i][j][0],grid[i][j][1],1]).reshape(-1,1)
            o_h=np.matmul(V,data)
            a_h=sigmoid(o_h)
            a_h=np.append(a_h,1).reshape(-1,1)
            o_l=np.matmul(W,a_h)
            a_l=sigmoid(o_l)
            grid_info[i][j]=a_l**2
    points=[]
    for i in range(grid_info.shape[1]):
        column=grid_info[:,i]
        pos=np.argmin(column)
        test1=grid[pos, i]
        points.append(np.array(grid[pos,i]))

    points=np.array(points)


    plt.plot(classA[0], classA[1], marker='o', color='red', ls='')
    plt.plot(classB[0], classB[1], marker='x', color='blue', ls='')
    plt.plot(points[:,0],points[:,1], color='green')
    plt.xlabel('Data x value')
    plt.ylabel('Data y value')
    plt.show()







if __name__ == '__main__':

    #print(label)
    eta=0.001
    #W=delta_learning1a1(data)
    #W,errorList=delta_learning_Batch(X,label,eta)
    #W=perceptron_batch(data)
    #W=perceptron1a1(data)
    #W=delta_learning1a1_NO_Bias(data)
    #W,errorList=delta_learning_Batch_NO_Bias(data[0:2,:],data[2].reshape((1,data.shape[1])),eta)
    #W=perceptron1a1_NO_Bias(data)
    #print(W[0].size)
    data=data[:,:int(0.7*data.shape[1])]
    val=data[:,int(0.7*data.shape[1]):]
    #V,W,errorList,error_train=nonlinear_delta_No_Bias(data,8,1,val)
    #V, W, errorList1 = nonlinear_delta_Bias_Seq(data, 20, 1, val)
    V, W, errorList2, error_train = nonlinear_delta_Bias(data, 8, 1, val)
    #for i in range(autoencoder_data.__len__()):
    #    t=random.randint(0,7)
    #    autoencoder_data[i][0][t]=1
    #val=autoencoder_data[int(0.7*autoencoder_data.__len__()):]
    #autoencoder_data=autoencoder_data[:int(0.7*autoencoder_data.__len__())]

    #V,W,errorList=autoencoder(autoencoder_data,3,8,val)


    #errorPlot(np.array(errorList1, dtype=np.float32)/val.shape[1])#/val.shape[1])
    #errorPlot(np.array(errorList2))
    #errorPlot(np.array(error_train,dtype=np.float32)/data.shape[1])
    #if W[0].size==2:
    #    a=np.append(W,[0])
     #   W=[a]


    #print(W)
    #W[0]=0*W[0]

    #print([(classA[0][i],classA[1][i]) for i in range(0,classA.shape[1])])
    #plt.plot([(classA[0][i],classA[1][i]) for i in range(0,classA.shape[1])], color='red')
    #Plotting(classA,classB,W[0])
    Plotting_2_layer(classA,classB,V,W,grid)
