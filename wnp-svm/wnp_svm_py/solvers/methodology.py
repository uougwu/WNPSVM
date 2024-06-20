
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import math
from kneed import KneeLocator
import matplotlib.transforms as mtransforms
import tensorflow as tf



##......................................................
def WNPSVM0(A,B,ay0,by0,Ds,epsa,epsb,delta):
    while True:
        ea = ay0
        am0 = np.dot(A,ay0)
        Da = delta*Ds
        Ga = np.dot(np.dot(A.T,np.diag((am0**2 + epsa**2)**(-1/2))),A)
        da = np.dot(B.T, np.sign(np.dot(B,ay0)))
        Fa = Ga + Da
        ya = np.linalg.solve(Fa,da)/(np.dot(da.T,np.dot(np.linalg.inv(Fa),da)))
        ay0 = ya

        eb = by0
        bm0 = np.dot(B,by0)
        Db = delta*Ds
        Gb = np.dot(np.dot(B.T,np.diag((bm0**2 + epsb**2)**(-1/2))),B)
        db = np.dot(A.T, np.sign(np.dot(A,by0)))
        Fb = Gb + Db
        yb = np.linalg.solve(Fb,db)/(np.dot(db.T,np.dot(np.linalg.inv(Fb),db)))
        by0 = yb

        if (np.linalg.norm(yb - eb)/np.linalg.norm(eb))<1e-6 and (np.linalg.norm(ya - ea)/np.linalg.norm(ea))<1e-6:
            break

    w1 = ya[0:A.shape[1]-1]
    w2 = yb[0:A.shape[1]-1]
    b1 = ya[A.shape[1]-1]
    b2 = yb[A.shape[1]-1]

    return w1, w2, b1, b2

##......................................................
def WNPSVM(A,B,ay0,by0,q,s,epsa,epsb,delta):
    while True:
        ea = ay0
        am0 = np.dot(A,ay0)
        Da = np.dot(delta,np.diag((ay0**2 + epsa**2)**((q-2)/s)))
        Ga = np.dot(np.dot(A.T,np.diag((am0**2 + epsa**2)**(-1/2))),A)
        da = np.dot(B.T, np.sign(np.dot(B,ay0)))
        Fa = Ga + Da
        ya = np.linalg.solve(Fa,da)/(np.dot(da.T,np.dot(np.linalg.inv(Fa),da)))
        ay0 = ya

        eb = by0
        bm0 = np.dot(B,by0)
        Db = np.dot(delta,np.diag((by0**2 + epsb**2)**((q-2)/s)))
        Gb = np.dot(np.dot(B.T,np.diag((bm0**2 + epsb**2)**(-1/2))),B)
        db = np.dot(A.T, np.sign(np.dot(A,by0)))
        Fb = Gb + Db
        yb = np.linalg.solve(Fb,db)/(np.dot(db.T,np.dot(np.linalg.inv(Fb),db)))
        by0 = yb

        if (np.linalg.norm(yb - eb)/np.linalg.norm(eb))<1e-6 and (np.linalg.norm(ya - ea)/np.linalg.norm(ea))<1e-6:
            break

    w1 = ya[0:A.shape[1]-1]
    w2 = yb[0:A.shape[1]-1]
    b1 = ya[A.shape[1]-1]
    b2 = yb[A.shape[1]-1]

    return w1, w2, b1, b2


def WNPSVMpredict(X_test,w1,w2,b1,b2):
    y_pred = np.sign(np.abs(np.dot(X_test,w1) + np.dot(np.ones(X_test.shape[0]),b1)) - np.abs(np.dot(X_test,w2) + np.dot(np.ones(X_test.shape[0]),b2)))
    for i in range(0,len(y_pred)):
        if y_pred[i] == -1:
            y_pred[i] = 0
    return y_pred


def weighted_npsvm(A,B,ay0,by0,epsam,deltam,q,s,var):
    first_sol = []
    second_sol = []
    if var == 'delta':
        for delta in deltam:
            print('delta:', delta)
            while True:
                ea = ay0
                am0 = np.dot(A,ay0)
                Da = np.dot(delta,np.diag((ay0**2 + epsam**2)**((q-2)/s)))
                Ga = np.dot(np.dot(A.T,np.diag((am0**2 + epsam**2)**(-1/2))),A)
                da = np.dot(B.T, np.sign(np.dot(B,ay0)))
                Fa = Ga + Da
                ya = np.linalg.solve(Fa,da)/(np.dot(da.T,np.dot(np.linalg.inv(Fa),da)))
                ay0 = ya

                eb = by0
                bm0 = np.dot(B,by0)
                Db = np.dot(delta,np.diag((by0**2 + epsam**2)**((q-2)/s)))
                Gb = np.dot(np.dot(B.T,np.diag((bm0**2 + epsam**2)**(-1/2))),B)
                db = np.dot(A.T, np.sign(np.dot(A,by0)))
                Fb = Gb + Db
                yb = np.linalg.solve(Fb,db)/(np.dot(db.T,np.dot(np.linalg.inv(Fb),db)))
                by0 = yb

                if (np.linalg.norm(yb - eb)/np.linalg.norm(eb))<1e-4 and (np.linalg.norm(ya - ea)/np.linalg.norm(ea))<1e-4:
                    break
            za = ya / np.linalg.norm(ya)
            zb = yb / np.linalg.norm(yb)
            Aproj = np.sort(np.abs(np.dot(A, zb)))[::-1]
            print(max(Aproj))
            Bproj = np.sort(np.abs(np.dot(B, za)))[::-1]
            print(max(Bproj))
            first_sol.append(Aproj)
            second_sol.append(Bproj)

    elif var == 'epsilon':
        for epsa in epsam:
            print('epsilon:', epsa)
            while True:
                ea = ay0
                am0 = np.dot(A,ay0)
                Da = np.dot(deltam,np.diag((ay0**2 + epsa**2)**((q-2)/s)))
                Ga = np.dot(np.dot(A.T,np.diag((am0**2 + epsa**2)**(-1/2))),A)
                da = np.dot(B.T, np.sign(np.dot(B,ay0)))
                Fa = Ga + Da
                ya = np.linalg.solve(Fa,da)/(np.dot(da.T,np.dot(np.linalg.inv(Fa),da)))
                ay0 = ya

                eb = by0
                bm0 = np.dot(B,by0)
                Db = np.dot(deltam,np.diag((by0**2 + epsa**2)**((q-2)/s)))
                Gb = np.dot(np.dot(B.T,np.diag((bm0**2 + epsa**2)**(-1/2))),B)
                db = np.dot(A.T, np.sign(np.dot(A,by0)))
                Fb = Gb + Db
                yb = np.linalg.solve(Fb,db)/(np.dot(db.T,np.dot(np.linalg.inv(Fb),db)))
                by0 = yb

                if (np.linalg.norm(yb - eb)/np.linalg.norm(eb))<1e-4 and (np.linalg.norm(ya - ea)/np.linalg.norm(ea))<1e-4:
                    break
            za = ya / np.linalg.norm(ya)
            zb = yb / np.linalg.norm(yb)
            Aproj = np.sort(np.abs(np.dot(A, zb)))[::-1]
            print(max(Aproj))
            Bproj = np.sort(np.abs(np.dot(B, za)))[::-1]
            print(max(Bproj))
            first_sol.append(Aproj)
            second_sol.append(Bproj)

    return np.stack(first_sol,axis=1), np.stack(second_sol,axis=1)

def wnpsvm(A,B,ay0,by0,epsa,delta,q,s):
    k = 0
    while True:
        k = k + 1
        ea = ay0
        am0 = np.dot(A,ay0)
        Da = np.dot(delta,np.diag((ay0**2 + epsa**2)**((q-2)/s)))
        Ga = np.dot(np.dot(A.T,np.diag((am0**2 + epsa**2)**(-1/2))),A)
        da = np.dot(B.T, np.sign(np.dot(B,ay0)))
        Fa = Ga + Da
        ya = np.linalg.solve(Fa,da)/(np.dot(da.T,np.dot(np.linalg.inv(Fa),da)))
        ay0 = ya

        eb = by0
        bm0 = np.dot(B,by0)
        Db = np.dot(delta,np.diag((by0**2 + epsa**2)**((q-2)/s)))
        Gb = np.dot(np.dot(B.T,np.diag((bm0**2 + epsa**2)**(-1/2))),B)
        db = np.dot(A.T, np.sign(np.dot(A,by0)))
        Fb = Gb + Db
        yb = np.linalg.solve(Fb,db)/(np.dot(db.T,np.dot(np.linalg.inv(Fb),db)))
        by0 = yb

        if (np.linalg.norm(yb - eb)/np.linalg.norm(eb))<1e-4 and (np.linalg.norm(ya - ea)/np.linalg.norm(ea))<1e-4:
            break
    
    return ya,yb


tf.compat.v1.set_random_seed(10)
tf.random.set_seed(10)
def compare_baselines(N,runs,var1,var2):
    Kingry = scipy.io.loadmat('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/PreKingrynorm_min.mat')
    A = Kingry['A']
    L = Kingry['B']
    za = Kingry['ya'].flatten()
    zb = Kingry['yb'].flatten()

    ya = za/np.linalg.norm(za)
    yb = zb/np.linalg.norm(zb) 
    Aproj = np.abs(np.dot(A,yb))
    Lproj = np.abs(np.dot(L,ya))
    ind1 = Aproj.argsort()[::-1]
    ind2 = Lproj.argsort()[::-1]
    ALB = A[ind1,:]    
    ALN = L[ind2,:]
   
    ALLB = ALB[0:N,:]
    ALLN = ALN[0:N,:]

    print(max(Aproj))
    print(max(Lproj))

    seed = 10

    if var1 == 'Schu4':
            X = StandardScaler().fit_transform(ALLB[:,0:48].T)
    elif var1 == 'LVS':
            X = StandardScaler().fit_transform(np.concatenate((ALLB[:,0:24].T,ALLB[:,48:72].T)))
    y_train = np.concatenate((np.array([0]*24),np.array([1]*24)))

    X1 = X[0:24,:]
    X2 = X[24:48,:]
    X_train = np.concatenate((X1,X2),axis=0)

    met = 16
    acc = np.zeros((met,runs))

    AA = np.hstack((X1,np.ones((X1.shape[0],1))))
    BB = np.hstack((X2,np.ones((X1.shape[0],1))))
    n = AA.shape[1]
    delta = 1e-1
    ay0 = np.random.rand(n)
    by0 = ay0
    epsa = 1e-2
    epsb = epsa

    ns = N
    mn1 = [-np.ones(ns),2*np.ones(ns+1),-np.ones(ns)]
    offset = [-1,0,1]
    DD1 = diags(mn1,offset).toarray() 
    Ds1 = np.dot(DD1.T,DD1) + 3*np.eye(ns+1)

    mn2 = [np.ones(ns+1),-np.ones(ns)]
    offset = [0,1]
    DD2 = diags(mn2,offset).toarray() 
    Ds2 = np.dot(DD2.T,DD2) + 3*np.eye(ns+1)

    for j in range(0,runs):
        
        clfRF = RandomForestClassifier(random_state=seed).fit(X_train, y_train)
        clfLR = LogisticRegression(random_state=seed).fit(X_train, y_train) 
        clfSVM = svm.SVC(kernel='linear').fit(X_train, y_train)
        clfDT = tree.DecisionTreeClassifier(max_depth=2,random_state=seed).fit(X_train, y_train) 
        if var1 == 'Schu4':
            clfNB = GaussianNB(var_smoothing=1e-2).fit(X_train, y_train) 
        elif var1 == 'LVS':
            clfNB = GaussianNB(var_smoothing=1e-3).fit(X_train, y_train) 
        clfKNN = KNeighborsClassifier().fit(X_train, y_train) 
        clfAB = AdaBoostClassifier(random_state=seed).fit(X_train, y_train) 
        clfGB = GradientBoostingClassifier(random_state=seed).fit(X_train, y_train) 
        clfXGB = XGBClassifier(random_state=seed).fit(X_train, y_train) 
        
        clfANN = Sequential()
        clfANN.add(Dense(128,activation='relu'))
        clfANN.add(Dense(1,activation='sigmoid'))
        clfANN.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        clfANN.fit(x=X_train,y=y_train,epochs=40,batch_size=64,verbose=0)
        
        aw1,aw2,ab1,ab2 = WNPSVM0(AA,BB,ay0,by0,Ds1,epsa,epsb,delta)
        bw1,bw2,bb1,bb2 = WNPSVM0(AA,BB,ay0,by0,Ds2,epsa,epsb,delta)
        cw1,cw2,cb1,cb2 = WNPSVM(AA,BB,ay0,by0,2,2,epsa,epsb,delta)
        dw1,dw2,db1,db2 = WNPSVM(AA,BB,ay0,by0,1,1,epsa,epsb,delta)
        ew1,ew2,eb1,eb2 = WNPSVM(AA,BB,ay0,by0,1,2,epsa,epsb,delta)
        fw1,fw2,fb1,fb2 = WNPSVM(AA,BB,ay0,by0,0.1,2,epsa,epsb,delta)
        
        if var2 == 'Schu4':
            Xtest = np.concatenate((ALLN[:,0:6].T,ALLN[:,24:48].T))
        elif var2 == 'LVS':
            Xtest = np.concatenate((ALLN[:,0:6].T,ALLN[:,48:72].T))

        y_test = np.concatenate((np.array([0]*6),np.array([1]*24)))
        X_test = StandardScaler().fit_transform(Xtest)

        y_predRF = clfRF.predict(X_test)
        y_predLR = clfLR.predict(X_test)
        y_predSVM = clfSVM.predict(X_test)
        y_predDT = clfDT.predict(X_test)
        y_predNB = clfNB.predict(X_test)
        y_predKNN = clfKNN.predict(X_test)
        y_predAB = clfAB.predict(X_test)
        y_predGB = clfGB.predict(X_test)
        y_predXGB = clfXGB.predict(X_test)
        y_predANN = clfANN.predict(X_test)>0.5
        y_predWNPSVM1 = WNPSVMpredict(X_test,aw1,aw2,ab1,ab2)
        y_predWNPSVM2 = WNPSVMpredict(X_test,bw1,bw2,bb1,bb2)
        y_predWNPSVM3 = WNPSVMpredict(X_test,cw1,cw2,cb1,cb2)
        y_predWNPSVM4 = WNPSVMpredict(X_test,dw1,dw2,db1,db2)
        y_predWNPSVM5 = WNPSVMpredict(X_test,ew1,ew2,eb1,eb2)
        y_predWNPSVM6 = WNPSVMpredict(X_test,fw1,fw2,fb1,fb2)

        acc[0,j] = balanced_accuracy_score(y_test, y_predRF)
        print('Random Forest:', balanced_accuracy_score(y_test, y_predRF))
        print(confusion_matrix(y_test, y_predRF))

        acc[1,j] = balanced_accuracy_score(y_test, y_predLR)
        print('Logistic Regression:', balanced_accuracy_score(y_test, y_predLR))
        print(confusion_matrix(y_test, y_predLR))

        acc[2,j] = balanced_accuracy_score(y_test, y_predSVM)
        print('SVM:', balanced_accuracy_score(y_test, y_predSVM))
        print(confusion_matrix(y_test, y_predSVM))

        acc[3,j] = balanced_accuracy_score(y_test, y_predDT)
        print('Decision Tree:', balanced_accuracy_score(y_test, y_predDT))
        print(confusion_matrix(y_test, y_predDT))

        acc[4,j] = balanced_accuracy_score(y_test, y_predNB)
        print('Naive Bayes:', balanced_accuracy_score(y_test, y_predNB))
        print(confusion_matrix(y_test, y_predNB))

        acc[5,j] = balanced_accuracy_score(y_test, y_predKNN)
        print('KNN:', balanced_accuracy_score(y_test, y_predKNN))
        print(confusion_matrix(y_test, y_predKNN))

        acc[6,j] = balanced_accuracy_score(y_test, y_predAB)
        print('Adaboost:', balanced_accuracy_score(y_test, y_predAB))
        print(confusion_matrix(y_test, y_predAB))

        acc[7,j] = balanced_accuracy_score(y_test, y_predGB)
        print('Gradientboost:', balanced_accuracy_score(y_test, y_predGB))
        print(confusion_matrix(y_test, y_predGB))

        acc[8,j] = balanced_accuracy_score(y_test, y_predXGB)
        print('Xgradientboost:', balanced_accuracy_score(y_test, y_predXGB))
        print(confusion_matrix(y_test, y_predXGB))
        
        acc[9,j] = balanced_accuracy_score(y_test, y_predANN)
        print('ANN:', balanced_accuracy_score(y_test, y_predANN))
        print(confusion_matrix(y_test, y_predANN))
        
        acc[10,j] = balanced_accuracy_score(y_test, y_predWNPSVM1)
        print('$\ell_1$-WNPSVM ($M = \mathcal{L}_1$):', balanced_accuracy_score(y_test, y_predWNPSVM1))
        print(confusion_matrix(y_test, y_predWNPSVM1))

        acc[11,j] = balanced_accuracy_score(y_test, y_predWNPSVM2)
        print('$\ell_1$-WNPSVM ($M = \mathcal{L}_2$):', balanced_accuracy_score(y_test, y_predWNPSVM2))
        print(confusion_matrix(y_test, y_predWNPSVM2))

        acc[12,j] = balanced_accuracy_score(y_test, y_predWNPSVM3)
        print('$\ell_1$-WNPSVM ($M = I$):', balanced_accuracy_score(y_test, y_predWNPSVM3))
        print(confusion_matrix(y_test, y_predWNPSVM3))

        acc[13,j] = balanced_accuracy_score(y_test, y_predWNPSVM4)
        print('$\ell_1$-WNPSVM ($M = D_{\epsilon,1}(\mathbf{z})^2$):', balanced_accuracy_score(y_test, y_predWNPSVM4))
        print(confusion_matrix(y_test, y_predWNPSVM4))

        acc[14,j] = balanced_accuracy_score(y_test, y_predWNPSVM5)
        print('$\ell_1$-WNPSVM ($M = D_{\epsilon,1}(\mathbf{z})$):', balanced_accuracy_score(y_test, y_predWNPSVM5))
        print(confusion_matrix(y_test, y_predWNPSVM5))

        acc[15,j] = balanced_accuracy_score(y_test, y_predWNPSVM6)
        print('$\ell_1$-WNPSVM ($M = D_{\epsilon,0.1}(\mathbf{z})$):', balanced_accuracy_score(y_test, y_predWNPSVM6))
        print(confusion_matrix(y_test, y_predWNPSVM6))

        indd = random.sample(range(0,24),24) # using 80%
        AA = AA[indd,:]
        BB = BB[indd,:]
        X_train = np.concatenate((X1[indd,:],X2[indd,:]),axis=0)

        print(j)

    print('Random Forest:', sum(acc[0,:])/runs,'Random Forest:', np.std(acc[0,:]))
    print('Logistic Regression:', sum(acc[1,:])/runs,'Logistic Regression:', np.std(acc[1,:]))
    print('SVM:', sum(acc[2,:])/runs,'SVM:', np.std(acc[2,:]))
    print('Decision Tree:', sum(acc[3,:])/runs,'Decision Tree:', np.std(acc[3,:]))
    print('Naive Bayes:', sum(acc[4,:])/runs,'Naive Bayes:', np.std(acc[4,:]))
    print('KNN:', sum(acc[5,:])/runs,'KNN:', np.std(acc[5,:]))
    print('Adaboost:', sum(acc[6,:])/runs,'Adaboost:', np.std(acc[6,:]))
    print('Gradientboost:', sum(acc[7,:])/runs,'Gradientboost:', np.std(acc[7,:]))
    print('Xgradientboost:', sum(acc[8,:])/runs,'Xgradientboost:', np.std(acc[8,:]))
    print('ANN:', sum(acc[9,:])/runs,'ANN:', np.std(acc[9,:]))
    print('$\ell_1$-WNPSVM ($M = \mathcal{L}_1$):', sum(acc[10,:])/runs,'$\ell_1$-WNPSVM ($M = \mathcal{L}_1)$:',
        np.std(acc[10,:]))
    print('$\ell_1$-WNPSVM ($M = \mathcal{L}_2$):', sum(acc[11,:])/runs,'$\ell_1$-WNPSVM ($M = \mathcal{L}_2)$:',
        np.std(acc[11,:]))
    print('$\ell_1$-WNPSVM ($\ell_1$-WNPSVM ($M = I$):', sum(acc[12,:])/runs,'$\ell_1$-WNPSVM ($M = I$):', np.std(acc[12,:]))
    print('$\ell_1$-WNPSVM ($M = D_{\epsilon,1}(\mathbf{z})^2$):', sum(acc[13,:])/runs,'$M = D_{\epsilon,1}(\mathbf{z})^2$:',
        np.std(acc[13,:]))
    print('$\ell_1$-WNPSVM ($M = D_{\epsilon,1}(\mathbf{z})$):', sum(acc[14,:])/runs,'$M = D_{\epsilon,1}(\mathbf{z})$:',
        np.std(acc[14,:]))
    print('$\ell_1$-WNPSVM ($M = D_{\epsilon,0.1}(\mathbf{z})^2$):', sum(acc[15,:])/runs,'$M = D_{\epsilon,0.1}(\mathbf{z})$:',
        np.std(acc[15,:]))

    c = ['r','b','k','y','m','g','sienna','purple','darkorange','deeppink','lime','darkviolet','slategrey','indigo','maroon',
        'olive']
    methods = ['Random Forest','Logistic Regression','Support Vector Machine','Decision Tree',
        'Naive Bayes','K-Nearest Neighbors','Adaptive Boosting','Gradient Boosting','Extreme Gradient Boosting',
            'Artificial Neural Network',
            '$\ell_1$-WNPSVM ($M = \mathcal{L}_1$)','$\ell_1$-WNPSVM ($M = \mathcal{L}_2$)','$\ell_1$-WNPSVM ($M = I$)',
            '$\ell_1$-WNPSVM $(M=D_{\epsilon,1}(\mathbf{z})^2)$','$\ell_1$-WNPSVM ($M=D_{\epsilon,1}(\mathbf{z})$)',
            '$\ell_1$-WNPSVM ($M=D_{\epsilon,0.1}(\mathbf{z})$)']
    fig = plt.figure(figsize = (12,8))
    for i in range(acc.shape[0]):
        plt.plot(np.sort(acc[i,:]),'--o',lw=5,label=methods[i],color=c[i])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
            title='Train: LVS $~~$ Test: LVS',fontsize=16)
    plt.ylabel('Sorted Balanced Accuracy Scores',fontsize=20)
    plt.xlabel('Index',fontsize=20)
    plt.tight_layout()
    ####plt.savefig('Sorted_Bal_Acc_LVS_LVS.jpg')
    plt.show()

    methods = ['Random Forest','Logistic Regression','Support Vector Machine','Decision Tree',
           'Naive Bayes','K-Nearest Neighbors','Adaptive Boosting','Gradient Boosting','Extreme Gradient Boosting','Artificial Neural Network',
              '$\ell_1$-WNPSVM ($M = \mathcal{L}_1$)','$\ell_1$-WNPSVM ($M = \mathcal{L}_2$)','$\ell_1$-WNPSVM ($M = I$)',
              '$\ell_1$-WNPSVM $(M=D_{\epsilon,1}(\mathbf{z})^2)$','$\ell_1$-WNPSVM ($M=D_{\epsilon,1}(\mathbf{z})$)',
               '$\ell_1$-WNPSVM ($M=D_{\epsilon,0.1}(\mathbf{z})$)']
    ss = np.zeros(acc.shape[0])
    fig = plt.figure(figsize = (12,8))
    for i in range(acc.shape[0]):
        ss[i] = sum(acc[i,:])/runs
    plt.plot(ss,'--o',color='b',lw=3)
    plt.xticks(range(len(ss)),methods,rotation=90,fontsize=20)
    plt.ylabel('Avg. Bal. Acc. Scores',fontsize=20)
    plt.xlabel('Methods',fontsize=20)
    plt.tight_layout()
    ##plt.savefig('Sorted_Bal_Acc_Avg_LVS.jpg')
    plt.show()
