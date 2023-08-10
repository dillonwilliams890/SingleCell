# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:13:39 2022

@author: will6605
"""
#%%
import cv2 as cv
from tracker_sat import *
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


def P50(oxy, sats):
    xData = oxy
    yData = sats
    # these are the same as the scipy defaults
    initialParameters = np.array([1.0, 1.0, 1.0])
    # do not print unnecessary warnings during curve_fit()
    warnings.filterwarnings("ignore")
    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, xData, yData, initialParameters)
    modelPredictions = func(xData, *fittedParameters) 
    absError = modelPredictions - yData
    xModel = np.linspace(min(xData), max(xData),num=1000)
    yModel = func(xModel, *fittedParameters)
    f = np.full((1000, ), 0.5)
    x = np.linspace(min(xData), max(xData),num=1000)
    g = func(xModel, *fittedParameters)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    p50=x[idx[0]]
    return p50

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
    frame2 = (raw[key][1])  # get the second frame
    #need a control frame under both flickering LEDs
    control_1 = frame1
    control_1[control_1 < 1] = 1
    control_2 = frame2
    control_2[control_2 < 1] = 1
    #The brighter frame in the 430nm frame the dimmer is 410
    if np.mean(control_1) > np.mean(control_2):
        control_430=control_1
        control_410=control_2
    else:
        control_430=control_2
        control_410=control_1
    
    #Set background removal object for later
    object_detector = cv.createBackgroundSubtractorMOG2()
    circ=[]
    deform=[]
    parea=5.6 #camera pixel area
    #Molecular absorbtion coefficints of something like that ~chemistry~
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)
    cell_img=[]
    #this is the loop that populates a list with np arrays of all the frames (this is the slow part)
    extinct=[]
    #cell_img=[]
    LED=[0,0]
    light = False #boolean, true for 430nm, false for 410, should switch every frame is hardware is correct
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
        LED.append(np.mean(frame[int(r3[1]):int(r3[1]+r3[3]), int(r3[0]):int(r3[0]+r3[2])]))
        #Skip frames if flickering goes wrong
        if LED[p]<5:
               p=p+2
               LED.append(LED[p-3])
        elif np.abs(LED[p]-LED[p-1])<3:
                p=p+1
        #If no anaylze the frame
        else:
            if p >= 4:
                #Normalize the frame with control frames
                if LED[p]>LED[p-1]:
                    light = True
                    img=(roi/control_430)      
                else:
                    light = False
                    img=(roi/control_410)      
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
                    area = cv.contourArea(hull, True)
                    circ = 2*np.sqrt(np.pi*area)/perimeter
                    rect = cv.minAreaRect(cnt)
                    x, y, w, h = cv.boundingRect(cnt) 
                    (cx, cy), (width, height), angle = rect
                    # box = cv.boxPoints(rect)
                    # box = np.int0(box)
                    minor=min(width, height)
                    major=max(width, height)
                    #(xx,yy),(minor,major),angle = cv.fitEllipse(cnt)
                    # if angle < 20 or angle > 160:
                    #     EI=(major-minor)/major
                    # else:
                    #     EI=(minor-major)/major
                    detections.append([x, y, w, h, circ, area, light,  major, minor, angle])

            # 2. Object Tracking, basic distance tracker takes in countour info and tracks moving cells
            boxes_ids = tracker.update(detections)
            if len(boxes_ids)>0:
                vol=np.zeros((boxes_ids[-1][4]+1000, 1))
            else:
                vol=[]
               
            for box_id in boxes_ids:
                #pts=[]
                Hb=[]
                x, y, w, h, id, circ, cx, area, light, major, minor, angle = box_id
                # cimg = np.zeros_like(img)
                vol[id]=np.add(vol[id], area)
                #Put a box around the cell
                Hb=(img[int(y-10):int(y+60), int(x-10):int(x+80)])
                Hb[Hb <= 0] = 0.01
                extinct.append([0])
                #for the given cell calcuate the light absorbtion in that box/frame
                extinct[id].append(parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hb)))))
                
                #Use boolean light to know which light was on during that frame and get that frames absorbtion and the provious one to calculate saturation
                if light == True:
                    f=extinct[id][-1] #410
                    e=extinct[id][-2] #430
                else:
                    e=extinct[id][-1] #410
                    f=extinct[id][-2] #430

                #Set absorbtion values to equation constants
                a=w410_d
                b=w410_o
                c=w430_d
                d=w430_o
                
                #Calcuate mass of oxygenated and deoxygenated hemoglobin
                Mo=(a*f-e*c)/(a*d-b*c)
                Md=(e*d-b*f)/(a*d-b*c)
                
                #Record saturation is cell is the the roi
                if np.max(vol)>4500:
                    saturation=10000
                    Hgb=-1
                elif cx < int(r1[2]+r1[0]) and cx > int(r2[0]) and abs(Mo) > 0 and abs(Md) > 0:
                    saturation = Mo/(Mo+Md)
                    cell_img=[]#roi[int(y):int(y+h), int(x):int(x+w)]
                    #print(cell_img)
                    Hgb=(Mo+Md)
                else:
                    saturation = -1
                    Hgb=-1
                #print(saturation)
                #Save circularity, saturation, area, and location for a given cell tagged to it's specific id
                deform.append([oxy, id, circ, cx, p, area, saturation, cell_img, Hgb, major, minor, angle])
                
            # ## For debugging ###
            #     EI=major/minor
            #     cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            #     cv.drawContours(roi, [hull], -1, (255, 255, 255), 1)
            #     # cv.putText(roi, str('%f' %EI), (x, y - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            #     # cv.putText(roi, str('%f' %angle), (x, y + 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
              
            # cv.imshow("roi", roi)
            # cv.waitKey(10)

        
            p=p+1
    return deform

#This function takes the cell info from previous function and calculates paramaters, not going to comment too much since it basically just puts info from previous function into dataframes
def shape(deform, r1, r2, r3, oxy):
    print(deform)
    value=[]
    df = pd.DataFrame (deform, columns = ['oxy', 'cell', 'circ', 'cx','time','area','saturation','cell_img','hemoglobin', 'major', 'minor', 'angle'])
    cells = {k: v for k, v in df.groupby('cell')}
    
    #Go through the info of each cell and calcuate total deformation and median saturation
    for i in range(len(cells)):
        tau=0
        tau_v=0
        dub = False
        sat=[]
        hgb=[]
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
        cell_steady=[]
        transit=[]
        angle=[]
        location=[]
        for index, c in cells[i].iterrows():
            if c['saturation']>10:
                dub=True
            elif c['saturation']>-0.8 and c['saturation']<1.5:
                sat.append(c['saturation'])
            if c['hemoglobin']>-0.5 and c['hemoglobin']<1.5:
                hgb.append(c['hemoglobin'])
                cell=c['cell_img']
            if c['cx'] > int(r1[0]+r1[2]): #not using this one right now
                    rest.append(c['circ'])
            if c['cx'] > int(r1[0]) and c['saturation']>-0.8: #add circulatiry and EI if the cell was in the squeeze section
                    squeeze.append(c['circ'])
                    squeezemajor.append(c['major'])
                    squeezeminor.append(c['minor'])
                    transit.append(c['time'])
                    location.append(c['cx'])
                    cell_steady.append(c['area'])
            if c['cx'] < int(r2[2]) and c['cx'] > int(r2[0]) and c['saturation']>-0.8: #add circularity and EI if cell was after squeeze
                    recovery.append(c['circ'])
                    recoverymajor.append(c['major'])
                    recoveryminor.append(c['minor'])
                    angle.append(c['angle'])
            # if c['cx'] < int(r1[0]+r1[2]/3) and c['cx'] > int(r2[0]) and c['saturation']>-0.8: #add circularity and EI if cell was after squeeze
            if  c['cx'] > int(r2[0]) and c['saturation']>-0.8: #add circularity and EI if cell was after squeeze
                    tauparm.append(c['circ'])
                    taumajor.append(c['major'])
                    tauminor.append(c['minor'])
                    dt.append(c['time'])
                    dx.append(c['cx'])
        if  dub==False and len(squeeze) >= 4 and len(recovery) >=5 and len(sat)>=5: #make sure cell was sampled enough to give good data
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
            speed=-140*(10**-6)*(location[-1]-location[0])/(transit[-1]-transit[0])
            # area=.0196*cell_steady
            area=.0196*(np.median(cell_steady))
            #w=sum(width)/np.count_nonzero(width)
            satu=np.median(sat)
            hemo=np.median(hgb)
            e=2*(np.sqrt(area/np.pi))/(6.5) #confinment parameter
            Ca=(speed*(10**-6))*5750
            value.append([oxy,D_Rc,Ta,area,speed,e,degrees,satu,cell,Ca,tau,tau_v])
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

    data3['sat_norm']=(data3['saturation']-np.mean(data3.loc[data3['oxy'] == 0]['saturation']))
    data3['sat_norm']=(data3['sat_norm'])/np.mean((data3.loc[data3['oxy'] == 21]['sat_norm']))    

    SO2=[np.mean((data3.loc[data3['oxy'] == 0]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 2]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 3]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 4]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 5]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 7]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 12]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 21]['sat_norm']))]
    #data3['def']=data3['Ta']/(data3['area']*data3['speed'])
    data3 = data3[data3['Ta'] < 0.7]
    data3 = data3[data3['Ta'] >= 0]
    data3 = data3[data3['sat_norm'] < 1.1]
    data3['D'] = data3['Ta']
    oxy=[0,2,3,4,5,7,12,21]
    # oxy=[2,4,6,12,21]
    N=len(oxy)
    LPF = []
    num_polys=[]
    num_solys=[]
    polys = pd.DataFrame()
    solys = pd.DataFrame()
    sort=pd.DataFrame()
    for i in range(len(oxy)):
        df = data3.loc[data3['oxy'] == oxy[i]][['oxy','D','sat_norm','area','speed']]
        data = data3.loc[data3['oxy'] == oxy[i]][['Ta','sat_norm']]

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
                if (j[1]['D'].mean())>0.2:
                    solys=pd.concat([solys, j[1]])
                else:
                    polys=pd.concat([polys, j[1]]) 


        else:
            if (df['D'].mean())>0.2:
                solys=pd.concat([solys, df])
            else:
                polys=pd.concat([polys, df]) 
        df.plot.scatter('D', 'sat_norm', c='cl', colormap='gist_rainbow')        
    solys['sort']=0
    polys['sort']=1
    polys=fit(polys, 'speed', 'D')
    polys=fit(polys, 'area', 'D')
    solys=fit(solys, 'speed', 'D')
    solys=fit(solys, 'area', 'D')
    SO2s=[0,np.mean((solys.loc[solys['oxy'] == 2]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 3]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 4]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 5]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 7]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 12]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 21]['sat_norm']))]
    SO2p=[np.mean((polys.loc[polys['oxy'] == 0]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 2]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 3]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 4]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 5]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 7]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 12]['sat_norm'])),1]
    SO2sstd=[0,np.std((solys.loc[solys['oxy'] == 2]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 3]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 4]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 5]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 7]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 12]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 21]['sat_norm']))]
    SO2pstd=[np.std((polys.loc[polys['oxy'] == 0]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 2]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 3]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 4]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 5]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 7]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 12]['sat_norm'])),1]
    oxys = oxy
    oxys = oxy
    oxyp= oxy
    p50t=P50(oxy,SO2)
    soly_nans = np.isnan(SO2s)
    SO2s=[d for (d, remove) in zip(SO2s, soly_nans) if not remove]
    oxys=[d for (d, remove) in zip(oxys, soly_nans) if not remove]
    poly_nans = np.isnan(SO2p)
    SO2p=[d for (d, remove) in zip(SO2p, poly_nans) if not remove]
    oxyp=[d for (d, remove) in zip(oxyp, poly_nans) if not remove]
    p50s=P50(oxys,SO2s)
    p50p=P50(oxyp,SO2p)
    p50=[p50s, p50p, p50t]
    LPF=[]
    sort=pd.concat([polys,solys])
    for i in range(len(oxy)):
        if len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1]) > 0:
            LPF.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1])/len(sort.loc[sort['oxy'] == oxy[i]]))
            num_polys.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1]))
            num_solys.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==0]))
        else:
            LPF.append(0)

    return data3, sort, SO2, LPF, polys, solys, SO2s, SO2p, p50, SO2sstd, SO2pstd, num_polys, num_solys

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
    datas = pd.DataFrame (data, columns = ['oxy', 'D_Rc', 'Ta', 'area','speed','e','angle','saturation','cell_img','Ca','tau','tau_v'])
    datas = datas[np.abs(datas['saturation']-datas['saturation'].mean()) <= (2*datas['saturation'].std())]    
    return datas

#Enter paths to speific videos and run program
#%%
# path=('D:/Single_Cell/xx/')  #path to video

oxygens =['0','2','3','4','5','7','12','21']
files =['data0.h5','data2.h5', 'data3.h5', 'data4.h5', 'data5.h5', 'data7.h5', 'data12.h5', 'data21.h5']

# oxygens =['2','2','4','4','6','6','12','12','21','21']
# files =[ 'data02.h5', 'data02_2.h5','data015.h5', 'data015_2.h5', 'data01.h5', 'data01_2.h5', 'data005.h5', 'data005_2.h5','data00.h5','data00_2.h5']


data=anaylsis(path, oxygens, files) 
data3=data.copy(deep=True)
#%%
clusters=[2,2,2,2,2,2,3,1]
data3['oxy'] = data3['oxy'].astype(int)
data3, sort, SO2, LPF, polys, solys, SO2s, SO2p, p50, SO2sstd, SO2pstd, num_polys, num_solys = fractions(data3, clusters) 
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
# data3 = pd.read_csv('C:/Users/will6605/Documents/SingleCell/Data2/20220930_CHC060.csv')
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
#%%
data3['kernel']=kernel
kern=data3[['oxy','sat_norm','D','kernel']]
kern.loc[kern['oxy'] == 2].to_clipboard(sep=',', index=False)  


# %%
# sort = sort.reset_index(drop=True)
values = np.vstack([sort.loc[sort['oxy'] == 21]["D"], sort.loc[sort['oxy'] == 21]["sat_norm"]])
kernel21 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 12]["D"], sort.loc[sort['oxy'] == 12]["sat_norm"]])
kernel12 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 7]["D"], sort.loc[sort['oxy'] == 7]["sat_norm"]])
kernel7 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 5]["D"], sort.loc[sort['oxy'] == 5]["sat_norm"]])
kernel5 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 4]["D"], sort.loc[sort['oxy'] == 4]["sat_norm"]])
kernel4 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 3]["D"], sort.loc[sort['oxy'] == 3]["sat_norm"]])
kernel3 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 2]["D"], sort.loc[sort['oxy'] == 2]["sat_norm"]])
kernel2 = stats.gaussian_kde(values)(values)
values = np.vstack([sort.loc[sort['oxy'] == 0]["D"], sort.loc[sort['oxy'] == 0]["sat_norm"]])
kernel0 = stats.gaussian_kde(values)(values)
kernel=np.concatenate((kernel0,kernel2,kernel3,kernel4,kernel5,kernel7,kernel12,kernel21),axis=0)
#%%

kern=sort[['oxy','sat_norm','D']]
# kern = kern.reset_index(drop=True)
kern['kernel']=kernel
kern.loc[kern['oxy'] == 2].to_clipboard(sep=',', index=False)  
# %%
