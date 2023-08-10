# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:13:39 2022

@author: will6605
"""
#%%
import cv2 as cv
from tracker_nosat import *
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import warnings
from scipy.ndimage import uniform_filter1d
#%%


def read_h5(file):
     with h5py.File(file, 'r') as f:
         frames = f['dataset'][()]
     return(frames)


def func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b)) 

def exponential_fit(x, eo, tau, einf):
    return eo * np.exp(-x/tau) + einf

def line_exp(x,uo,tau):
    return uo-x/tau


def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 2)
    upper_limit = s.mean() + (s.std() * 2)
    return ~s.between(lower_limit, upper_limit)

#Basic roi function to select the four rois necessary for processing
def get_roi(file):
    f = h5py.File(file, 'r')
    key = list(f.keys())[0]
    frame1 = (f[key][0])
    r1 = cv.selectROI(frame1) # squeeze section
    r2 = cv.selectROI(frame1) # sat measurement section
    r3 = cv.selectROI(frame1) # LED measurement section
    # print(r2)
    cv.destroyAllWindows()
    return [r1, r2, r3]

#Object tracking function that finds cells and passes info on their contours, location, and the saturation
#@profile
def track(file, r1, r2, r3, oxy):
    raw = h5py.File(file, 'r')
    key = list(raw.keys())[0]
    myset=raw['dataset']
    size=myset.shape[0]
    #raw_array = read_h5(file)
    #frames=[]
    tracker = EuclideanDistTracker() #distnace tracker that ids cells, from custom tracker_sat file
    frame1 = (raw[key][0]) # get the first frame
    #need a control frame under both flickering LEDs
    control = frame1
    control[control < 1] = 1
    #The brighter frame in the 430nm frame the dimmer is 410
    
    #Set background removal object for later
    object_detector = cv.createBackgroundSubtractorMOG2()
    circ=[]
    deform=[]
    parea=5.6 #camera pixel area
    cell_img=[]
    #this is the loop that populates a list with np arrays of all the frames (this is the slow part)
    p=2
    #While loop to iterate through the frames and do image processing
    while True:
        if p>=(size):
            break
        frame= (raw[key][p])
        if frame is None :
            break

        # Extract Region of interest
        roi = frame
        img = np.zeros_like(roi)
        #Get pixel intesity inLED roi to figure out which light is on
        if p >= 2:
            img=(roi/control) 
            mimg=(100*img)
            im = mimg.astype(np.uint8)
            mask = object_detector.apply(im)
            mask = cv.GaussianBlur(mask, (5, 5), 0)
            kernel = np.ones((5,5),np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
            # thresh= 255 - mask
            thresh=mask
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL ,cv.CHAIN_APPROX_NONE) 
            detections = []
            
            #Loop through the contours found in the image and record the ones of cells
        for cnt in contours:
                # Calculate area and remove small elements
                area = cv.contourArea(cnt)
                x, y, w, h = cv.boundingRect(cnt)   
                if area > 1000 and area < 4500 and x>r2[0]-50:
                    #Get perimiter and area to calculate circularity
                    hull = cv.convexHull(cnt)
                    perimeter = cv.arcLength(hull, True)  
                    # x, y, w, h = cv.boundingRect(cnt)
                    ellipse = cv.fitEllipse(cnt)   
                    (xc,yc),(d1,d2),angle = ellipse
                    area = cv.contourArea(hull, True)
                    circ = 2*np.sqrt(np.pi*area)/perimeter
                    rect = cv.minAreaRect(cnt)
                    x, y, w, h = cv.boundingRect(cnt) 
                    (cx, cy), (width, height), angle = rect
                    minor=min(width, height)
                    major=max(width, height)
                    detections.append([x, y, w, h, circ, area,  major, minor, angle,p])

            # 2. Object Tracking, basic distance tracker takes in countour info and tracks moving cells
        boxes_ids = tracker.update(detections)
        if len(boxes_ids)>0:
                vol=np.zeros((boxes_ids[-1][4]+1000, 1))
        else:
                vol=[]
               
        for box_id in boxes_ids:
                #pts=[]
                x, y, w, h, id, circ, cx, area, major, minor, angle, p = box_id
                # cimg = np.zeros_like(img)
                vol[id]=np.add(vol[id], area)
                if np.max(vol)>5000:
                    circ=100
                #Put a box around the cell
                # cell_img=[]
                if x>int(r2[0]) and x<int(r2[0])+100:
                    # print(x)
                    # print(cx)
                    cell_img=roi[int(y):int(y+h), int(x):int(x+w)]
                    # plt.imshow(cell_img, interpolation='none')
                    # plt.show()
                #print(saturation)
                #Save circularity, saturation, area, and location for a given cell tagged to it's specific id
                deform.append([oxy, id, circ, cx, p, area, cell_img, major, minor, angle,p])
            
                
            # ## For debugging ###
        #         EI=major/minor
        #         cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        #         cv.drawContours(roi, [hull], -1, (255, 255, 255), 1)
        #         # cv.putText(roi, str('%f' %EI), (x, y - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        #         cv.putText(roi, str('%f' %circ), (x, y + 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
              
        # cv.imshow("roi", roi)
        # cv.waitKey(0)

        
        p=p+1
    return deform

#This function takes the cell info from previous function and calculates paramaters, not going to comment too much since it basically just puts info from previous function into dataframes
def shape(deform, r1, r2, r3, oxy):
    value=[]
    df = pd.DataFrame (deform, columns = ['oxy', 'cell', 'circ', 'cx','time','area','cell_img', 'major', 'minor', 'angle','p'])
    cells = {k: v for k, v in df.groupby('cell')}
    
    #Go through the info of each cell and calcuate total deformation and median saturation
    for i in range(len(cells)):
        tau=0
        tau_v=0
        dub = False
        rest=[]
        squeeze=[]
        squeezemajor=[]
        squeezeminor=[]
        recovery=[]
        recoverymajor=[]
        recoveryminor=[]
        tau=np.nan
        tauparm=[]
        taumajor=[]
        tauminor=[]
        dx=[]
        dt=[]
        vx=[]
        t=[]
        tauEID=[]
        time=[]
        start_deform=[]
        cell_shear=[]
        cell_steady=[]
        transit=[]
        angle=[]
        location=[]
        for index, c in cells[i].iterrows():
            if c['circ']>10:
                dub=True
            if c['circ']>-0.5 and c['circ']<1.5:
                cell=c['cell_img']
            if c['cx'] > int(r1[0]+r1[2]): #not using this one right now
                    rest.append(c['circ'])
            if c['cx'] > int(r1[0]): #add circulatiry and EI if the cell was in the squeeze section
                    squeeze.append(c['circ'])
                    squeezemajor.append(c['major'])
                    squeezeminor.append(c['minor'])
                    transit.append(c['time'])
                    location.append(c['cx'])
                    cell_shear.append(c['area'])
                    p=c['p']
            if c['cx'] < int(r2[2]) and c['cx'] > int(r2[0]): #add circularity and EI if cell was after squeeze
                    recovery.append(c['circ'])
                    recoverymajor.append(c['major'])
                    recoveryminor.append(c['minor'])
                    angle.append(c['angle'])
                    cell_steady.append(c['area'])
            # if c['cx'] < int(r1[0]+r1[2]/3) and c['cx'] > int(r2[0]) and c['saturation']>-0.8: #add circularity and EI if cell was after squeeze
            if  c['cx'] > int(r2[0]): #add circularity and EI if cell was after squeeze
                    tauparm.append(c['circ'])
                    taumajor.append(c['major'])
                    tauminor.append(c['minor'])
                    dt.append(c['time'])
                    dx.append(c['cx'])
        if  dub==False and len(squeeze) >= 4 and len(recovery) >=5: #make sure cell was sampled enough to give good data
            #R=sum(rest)/np.count_nonzero(rest)
            D=np.median(squeeze)
            Rc=np.median(recovery)
            D_Rc=abs(D-Rc) #calculate change in circularity
            majorD=np.median(squeezemajor)
            minorD=np.median(squeezeminor)
            majorRC=np.median(recoverymajor)
            minorRC=np.median(recoveryminor)
            degrees=np.median(angle) 
            #calculate taylor deformation
            # if D_Rc < 0.06: #If low deformation then major and semimajor axis stay correct, undo axis switch
            EID=((majorD/2)-(minorD/2))/((majorD/2)+(minorD/2))
            EIRC=((majorRC/2)-(minorRC/2))/((majorRC/2)+(minorRC/2))
            eo=EID-EIRC
            einf=EIRC
            Ta=abs(EID-EIRC)
            # if len(tauminor) > 15:
            #     for j in range(len(dx)-1):
            #         vx.append(-140*(10**-6)*(dx[j+1]-dx[j]))
            #         t.append(j/1000)
            #     w=t
            #     z=uniform_filter1d(vx, size=3)
                
            #     # print(w)
            #     print(z)
            #     # q0 = (vx[0],.002,vx[-1]) # starting search koefs
            #     try:
            #         # opt, pcov = curve_fit(exponential_fit, w, z, q0)
            #         # vo, tau_v, vinf = opt
            #         # # test result
            #         # print(tau_v)
            #         # w2 = np.linspace(0, t[-1], 250)
            #         # z2 = exponential_fit(w2, vo, tau_v, vinf)
            #         fig, ax = plt.subplots()
            #         # ax.plot(w2, z2, color='k', label='Fit. func: $f(x) = %.3f e^{t/%.3f} %+.3f$' % (vo, tau_v, vinf))
            #         ax.plot(w, z, 'ro', label='data with noise')
            #         ax.legend(loc='best')
            #         plt.show()
            #     except RuntimeError:
            #         print("Error - curve_fit failed")
            #     for i in range(len(tauminor)-1):
            #         major=(taumajor[i])/(1+(z[i])*15)
            #         print(taumajor[i])
            #         print(major)
            #         print(tauminor[i])
            #         val=((major/2)-(tauminor[i]/2))/((major/2)+(tauminor[i]/2))
            #         if val > 0:
            #             tauEID.append(val)
            #         else:
            #             tauEID.append(0.001)
            #         time.append(i/1000)
            #     # print(time)
            #     print(tauEID)
            #     x=time
            #     y=uniform_filter1d(tauEID, size=3)
            #     # p0 = (eo,.002,einf) # starting search koefs
            #     try:
            #         # opt, pcov = curve_fit(exponential_fit, x, y, p0)
            #         # uo, tau, uinf = opt
            #         # # test result
            #         # print(tau)
            #         # x2 = np.linspace(0, time[-1], 250)
            #         # y2 = exponential_fit(x2, uo, tau, uinf)
            #         fig, ax = plt.subplots()
            #         # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{t/%.3f} %+.3f$' % (uo, tau, uinf))
            #         ax.plot(x, y, 'ko', label='data with noise')
            #         ax.legend(loc='best')
            #         plt.show()
            #     except RuntimeError:
            #         print("Error - curve_fit failed")
                # print(dx)
            #shear modulus calc
            mu=0.00826 #[kg/m-s]
            u=-140*2*(location[-1]-location[0])/(transit[-1]-transit[0]) #[um/s]
            # majorD=np.max(squeezemajor)
            # minorD=np.min(squeezeminor)
            # majorRC=np.min(recoverymajor)
            # minorRC=np.max(recoveryminor)
            w_d=majorD*.14 #[um]
            h_d=minorD*.14 #[um]
            w_rc=majorRC*.14 #[um]
            h_rc=minorRC*.14 #[um]
            thick=0.05 #[um]
            l=(w_d-w_rc)
            d=7.76-h_d
            d=.5
            shear=mu*(u/(d)) #[N/m^2]
            gamma=(np.arctan(l/(h_d/2)))
            G=thick*(shear/gamma) #[uN/m]
            poisson=0.5
            E=2*(shear/gamma)*(1+poisson)
            # print('Ta is ' +str(Ta))
            # print('u is ' +str(u))
            # print('w_d is ' +str(w_d))
            # print('h_d is ' +str(h_d))
            # print('w_rc is '+str(w_rc))
            # print('l is ' +str(l))
            # print('d is '+str(d))
            # print('shear is ' +str(shear))
            # print('gamma is ' +str(gamma))
            # print('G is ' +str(G))
            speed=u*(10**-6)
            # area=.0196*cell_steady
            area_shear=.0196*(np.median(cell_shear))
            area_steady=.0196*(np.median(cell_steady))
            e=(area_shear-area_steady)/area_steady
            #w=sum(width)/np.count_nonzero(width)
            # e=2*(np.sqrt(area/np.pi))/(6.5) #confinment parameter
            value.append([oxy,D_Rc,Ta,area_steady,u,l,cell,G,E,h_d, w_d, h_rc, w_rc])
    return value

#this function normalizes based on cell width
def fractions(data, cluster):
    def fit(data, x, y):
        X = data[x]#-(np.mean(data2.loc[data2['oxy'] == 0]['w'])) # here we have 2 variables for the multiple linear regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example
        Y = data[y]#+(np.mean(data2.loc[data2['oxy'] == 0]['D_Rc']))
        X = sm.add_constant(X) # adding a constant
        mean=np.mean(data[y])
        model = sm.OLS(Y, X).fit()
        coef=model.params[x]
        const=model.params.const
        data[y] = mean*data.apply(lambda row: ((row[y]))/(coef*(row[x])+const), axis=1)
        return data

    data3=data.copy(deep=True)

#data3['def']=data3['Ta']/(data3['area']*data3['speed'])
    data3 = data3[data3['Ta'] < 0.7]
    data3 = data3[data3['Ta'] >= 0]
    # data3 = data3[data3['e'] <= 1.4]
    data3['D'] = data3['D_Rc']
    # oxy=[0,2,3,4,5,7,12,21]
    oxy=[0]#,5]
    N=len(oxy)
    LPF = []
    polys = pd.DataFrame()
    solys = pd.DataFrame()
    sort=pd.DataFrame()
    for i in range(len(oxy)):
        df = data3.loc[data3['oxy'] == oxy[i]][['oxy','D', 'Ta','area','speed', 'G', 'E','h_d', 'w_d', 'h_rc', 'w_rc']]
        data = data3.loc[data3['oxy'] == oxy[i]][['Ta','D']]

        # Convert DataFrame to matrix
        mat = data.values
        # Using sklearn
        km = KMeans(n_clusters=cluster[i])
        km.fit(mat)
        # Get cluster assignment labels
        df['cl'] = km.labels_
        if cluster[i]>1:
            df1 = df[df['cl'] == 1]
            df2 = df[df['cl'] < 1]
            df3 = df[df['cl'] == 2]
            df4 = df[df['cl'] == 3]
            # df5 = df[df['cl'] == 4]
            j = [df1,df2,df3,df4]
            j=sorted(j, key=lambda x: x['D'].mean())
            if cluster[i] == 2:
                polys=pd.concat([polys, j[0]])
                solys=pd.concat([solys, j[1]])
            elif cluster[i] == 3:
                polys=pd.concat([polys, j[0]])
                solys=pd.concat([solys, j[2]])
                if (j[1]['D'].mean())>0.05:
                    solys=pd.concat([solys, j[1]])
                else:
                    polys=pd.concat([polys, j[1]]) 


        else:
            if (df['D'].mean())>0.05:
                solys=pd.concat([solys, df])
            else:
                polys=pd.concat([polys, df]) 
        df.plot.scatter('Ta', 'D', c='cl', colormap='gist_rainbow')        
    solys['sort']=0
    polys['sort']=1
    polys=fit(polys, 'speed', 'D')
    polys=fit(polys, 'area', 'D')
    solys=fit(solys, 'speed', 'D')
    solys=fit(solys, 'area', 'D')
    LPF=[]
    sort=pd.concat([polys,solys])
    for i in range(len(oxy)):
        if len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1]) > 0:
            LPF.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1])/len(sort.loc[sort['oxy'] == oxy[i]]))
        else:
            LPF.append(0)

    return data3, sort, LPF, polys, solys

#Function to run all the functions for a video
def process(file, r, oxy):
    deform = track(file, r[0], r[1], r[2], oxy)
    data=shape(deform, r[0], r[1], r[2], oxy) 
    return data

def anaylsis(path, oxygens, files):
    roi=[]
    data=[]
    for i in range(len(oxygens)):
        r=get_roi(path+files[i])
        roi.append(r)

    for i in range(len(oxygens)):
        value=process(path+files[i], roi[i], oxygens[i])
        data=data+value
    datas = pd.DataFrame (data, columns = ['oxy', 'D_Rc', 'Ta', 'area','speed','l','cell_img','G','E','h_d', 'w_d', 'h_rc', 'w_rc'])  
    return datas

#Enter paths to speific videos and run program
#%%
path=('F:/Single_Cell/20230425_CHC026/')  
# path=('C:/Users/will6605/Documents/SingleCell/20230117_CHC038/')
oxygens =['0']#,'5']#,'5','6']#,'5','7','12','21']
files =['data0f.h5']#,'data5f.h5']#, 'data5f.h5', 'data5new.h5']#, 'data5.h5', 'data7.h5', 'data12.h5', 'data21.h5']

# oxygens =['2','2','4','4','6','6','12','12','21','21']
# files =[ 'data02.h5', 'data02_2.h5','data015.h5', 'data015_2.h5', 'data01.h5', 'data01_2.h5', 'data005.h5', 'data005_2.h5','data00.h5','data00_2.h5']


data=anaylsis(path, oxygens, files) 
data3=data.copy(deep=True)
#%%
clusters=[3]#,2,1,1,1]
# clusters=[1,1,1,1,1,1,1,1]
data3['oxy'] = data3['oxy'].astype(int)
data3, sort, LPF, polys, solys = fractions(data3, clusters) 
# data3 = data3[~data3.groupby('oxy')['sat_norm'].apply(is_outlier)]

# %%
df = sort.loc[sort['oxy'] == 0].loc[sort['sort'] == 1]['D']
df1 = sort.loc[sort['oxy'] == 2].loc[sort['sort'] == 1]['D']
df2 = sort.loc[sort['oxy'] == 3].loc[sort['sort'] == 1]['D']
df3 = sort.loc[sort['oxy'] == 4].loc[sort['sort'] == 1]['D']
df4 = sort.loc[sort['oxy'] == 5].loc[sort['sort'] == 1]['D']
df5 = sort.loc[sort['oxy'] == 7].loc[sort['sort'] == 1]['D']
df.reset_index(drop=True, inplace=True)
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df4.reset_index(drop=True, inplace=True)
df5.reset_index(drop=True, inplace=True)
polysp=pd.concat([df,df1,df2,df3,df4,df5], ignore_index=True, axis=1)
polysp.to_clipboard(sep=',', index=False)  
# %%
#%%
######################################
# data3.to_csv(r'C:/Users/will6605/Documents/SingleCell/Data2/20221026_CHC026.csv',index=False)
# data3 = pd.read_csv('C:/Users/will6605/Documents/SingleCell/Data2/20221026_CHC026.csv')
oxy=[0,2,3,4,5,7,12,21]
for i in (oxy):
    print(np.mean(sort['Ta'].loc[sort['oxy'] == i]))

#%%
# sns.kdeplot(data=data, x='Ta',common_norm=False,hue='oxy', palette='coolwarm')
# plt.xlim(-0.15, 2)

c=['#cf453c', '#e46e56', '#f29274', '#f7af91', '#f3c7b1', '#e6d7cf', '#d3dbe7', '#bbd1f8', '#a1c0ff', '#86a9fc', '#6b8df0', '#516ddb']
fig, axes = plt.subplots(1, 8, figsize=(15, 5), sharey=True)
custom_xlim = (-0.001, .5)
custom_ylim = (-0.15,1.2)

# Setting the values for all axes.
plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

values = np.vstack([sort.loc[sort['oxy'] == 21]["D"], sort.loc[sort['oxy'] == 21]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[0], data=sort.loc[sort['oxy'] == 21],x='D', y='sat_norm', s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 12]["D"], sort.loc[sort['oxy'] == 12]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[1], data=sort.loc[sort['oxy'] == 12],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 7]["D"], sort.loc[sort['oxy'] == 7]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[2], data=sort.loc[sort['oxy'] == 7],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 5]["D"], sort.loc[sort['oxy'] == 5]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[3], data=sort.loc[sort['oxy'] == 5],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 4]["D"], sort.loc[sort['oxy'] == 4]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[4], data=sort.loc[sort['oxy'] == 4],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 3]["D"], sort.loc[sort['oxy'] == 3]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[5], data=sort.loc[sort['oxy'] == 3],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 2]["D"], sort.loc[sort['oxy'] == 2]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[6], data=sort.loc[sort['oxy'] == 2],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
values = np.vstack([sort.loc[sort['oxy'] == 0]["D"], sort.loc[sort['oxy'] == 0]["sat_norm"]])
kernel = stats.gaussian_kde(values)(values)
sns.scatterplot(ax=axes[7], data=sort.loc[sort['oxy'] == 0],x='D', y='sat_norm',  s= 5,c=kernel,cmap="viridis")
# df.loc[df['oxy'] == 7][['oxy','sat_norm','D','kernel']].to_clipboard(sep=',', index=False)  
# for ax in fig.axes: ax.axvline(40, 0,1,color='k', linestyle='--',linewidth=1)

# sns.kdeplot(ax=axes[0], data=data3.loc[data3['oxy'] == 21],x='Ta', y='sat_norm', fill=True, color=c[0],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[1], data=data3.loc[data3['oxy'] == 12],x='Ta', y='sat_norm', fill=True, color=c[2],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[2], data=data3.loc[data3['oxy'] == 7],x='Ta', y='sat_norm',  fill=True, color=c[5],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[3], data=data3.loc[data3['oxy'] == 5],x='Ta', y='sat_norm',  fill=True, color=c[6],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[4], data=data3.loc[data3['oxy'] == 4],x='Ta', y='sat_norm',  fill=True, color=c[7],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[5], data=data3.loc[data3['oxy'] == 3],x='Ta', y='sat_norm',  fill=True, color=c[8],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[6], data=data3.loc[data3['oxy'] == 2],x='Ta', y='sat_norm',  fill=True, color=c[9],levels=[0.1, 0.3, 0.6,1])
# sns.kdeplot(ax=axes[7], data=data3.loc[data3['oxy'] == 0],x='Ta', y='sat_norm',  fill=True, color=c[11],levels=[0.1, 0.3, 0.6,1])
# for ax in fig.axes: ax.axvline(20, 0,1,color='k', linestyle='--',linewidth=1)



# %%
values = np.vstack([data3.loc[data3['oxy'] == 21]["D"], data3.loc[data3['oxy'] == 21]["sat_norm"]])
kernel21 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 12]["D"], data3.loc[data3['oxy'] == 12]["sat_norm"]])
kernel12 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 7]["D"], data3.loc[data3['oxy'] == 7]["sat_norm"]])
kernel7 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 5]["D"], data3.loc[data3['oxy'] == 5]["sat_norm"]])
kernel5 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 4]["D"], data3.loc[data3['oxy'] == 4]["sat_norm"]])
kernel4 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 3]["D"], data3.loc[data3['oxy'] == 3]["sat_norm"]])
kernel3 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 2]["D"], data3.loc[data3['oxy'] == 2]["sat_norm"]])
kernel2 = stats.gaussian_kde(values)(values)
values = np.vstack([data3.loc[data3['oxy'] == 0]["D"], data3.loc[data3['oxy'] == 0]["sat_norm"]])
kernel0 = stats.gaussian_kde(values)(values)
kernel=np.concatenate((kernel0,kernel2,kernel3,kernel4,kernel5,kernel7,kernel12,kernel21),axis=0)
# test function
# def function(data, a, b, c):
#     x = data[0]
#     y = data[1]
#     return a * (x*b) + (y*c)

# Ta=np.array([0.0715,0.1392,0.029, 0.113,0.173, 0.227,0.283,0.069, 0.141,0.206, 0.259,0.31,0.133, 0.208,0.264, 0.316,0.368,0.217, 0.27,0.316, 0.358,0.406])
# e=np.array([0.893, 1.01,0.783, 0.897,1.011, 1.137,1.280,0.785, 0.9,1.023, 1.147,1.29,0.781, 0.907,1.036, 1.166,1.317,0.785, 0.916,1.046, 1.176,1.332])
# Ca=[0.2,0.2,0.3,0.3,0.3,0.3,0.3,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6]
# UcU=[1.228, 1.119,1.371, 1.272,1.165, 1.052,0.898,1.397, 1.3,1.202, 1.088,0.934,1.449, 1.363,1.265, 1.154,1.007,1.5, 1.416,1.315, 1.205,1.057]
# CaCell = np.array([(a) * (b) for a,b in zip(Ca, UcU)])
# x_data = e
# y_data = Ta
# z_data = CaCell

# # get fit parameters from scipy curve fit
# parameters, covariance = curve_fit(function, [x_data, y_data], z_data)

# # create surface function model
# # setup data points for calculating surface model
# model_x_data = np.linspace(min(x_data), max(x_data), 30)
# model_y_data = np.linspace(min(y_data), max(y_data), 30)
# # create coordinate arrays for vectorized evaluations
# X, Y = np.meshgrid(np.linspace(0.7, 1.5, num=30), np.linspace(-.1, 0.7, num=30))
# # calculate Z coordinate array
# Z = function(np.array([X, Y]), *parameters)

# # setup figure object

# # setup 3d object
# # ax = Axes3D(fig)
# # plot surface
# # fig, ax = plt.subplots()
# # c=ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=0, vmax=7)
# # ax.set_title('pcolormesh')
# # fig.colorbar(c, ax=ax)
# # ax.scatter(x_data,y_data,color='white')
# # ax.set_ylim([0,0.6])
# # sns.kdeplot(data=data, x='e',y='Ta', color='white',levels=5)
# # plt.show()


# # %%
# data3=data.copy(deep=True)
# data3['Ta'] = data3.apply(lambda row: (10**6)*(row['Ca']/((parameters[0]+(row['e'])*parameters[1])+((row['Ta'])*parameters[2]))), axis=1)
# # data3.to_csv(r'C:/Users/will6605/Documents/SingleCell/Data/20221102_CHC039.csv',index=False)
# # %%


# %%
# from matplotlib import pyplot as plt
# circ=polys.loc[polys['oxy'] == 5].sort_values('D_Rc')
# for index, row in polys.iterrows():
#     print(row['p'], row['D_Rc'])
#     plt.imshow(row['cell_img'], interpolation='none')
#     plt.show()
