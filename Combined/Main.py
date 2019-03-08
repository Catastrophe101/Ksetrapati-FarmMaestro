from sklearn import linear_model
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

model_list=[]
kmeans=None
df_train = pd.read_csv('Datasets/Dataset(Module-2).csv')
df_tr = df_train
def train_mod_one():
    spreadsheet = pd.ExcelFile('Datasets/Dataset(Module-1).xlsx')
    df2 = spreadsheet.parse('Sheet1')
    # df2.head()
    crops_combo=[]
    crop_season=[]
    for row in df2.itertuples():
        #use row[5] strip
        crop_season.append(str(row[5]).strip())
        #use row[4] strip
        crop_season.append(str(row[4]).strip())
        crops_combo.append(crop_season)
        crop_season=[]
    a = set(tuple(i) for i in crops_combo)
    # print('length of set :',len(a))
    # for l in sorted(a):
    #     print(l[0]," ",l[1])
    ar = []
    ar2d = []
    x_list = []
    z_list = []
    i = 0
    y_list = []
    for l in sorted(a):
        for row in df2.itertuples():
            if str(row[5]).strip() == l[0] and str(row[4]).strip() == l[1]:
                ar.append(row[8])
                x_list.append(row[8])
                ar.append(row[9])
                z_list.append(row[9])
                y_list.append(row[10])
                ar2d.append(ar)
                ar = []
        if len(ar2d) < 10:
            continue
        nar2d = np.array(ar2d)
        nar2d = np.nan_to_num(nar2d)
        ny = np.array(y_list)
        ny = np.nan_to_num(ny)
        X_train = nar2d[:-4]
        X_test = nar2d[-4:]
        y_train = ny[:-4]
        y_test = ny[-4:]
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X_train, y_train)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(X_test)
        model_list.append([ransac, l[0], l[1]])


def yieldPredict(test_input):
    y_predicted = []
    for m in model_list:
        ys = 10*m[0].predict(test_input)
        if ys[0]>0:
            y_predicted.append([ys[0], m[1], m[2]])
        # print(m[1], m[2])
        # print(ys)
    y_predicted.sort(reverse=True)
    df=pd.DataFrame(np.array(y_predicted),columns=['Current Yeild(q/ha)','Crop','Season'])
    # for i in y_predicted:
    #     print(i[0], " ", i[1], " ", i[2])
    return df

def train_mod_two():
    clmns = ['Available-N(kg/ha)', 'Available-P(kg/ha)', 'Available-K(kg/ha)']
    tr_clmns = ['Available-N(kg/ha)', 'Available-P(kg/ha)', 'Available-K(kg/ha)']
    i = 70
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df_tr[tr_clmns])
    labels = kmeans.labels_
    df_tr['clusters'] = labels
    clmns.extend(['clusters'])
    df_clust = df_tr[clmns]
    df_test = df_tr[tr_clmns]
    return kmeans


def varietyPredict(kmeans,values,out_one):
    prediction = kmeans.predict(values)
    out_second = df_tr[['Crop', 'Variety','Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)', 'Yeild(q/ha)']].loc[df_tr['clusters'] == prediction[0]]
    out_second=pd.merge(out_second,out_one,on='Crop',how='inner')
    print(out_second.to_string())
    return out_second


if __name__== "__main__":
    p=10
    print("Training module - 1")
    train_mod_one()
    print("Training module - 2")
    kmeans=train_mod_two()
    while p!=float(0):
        print("Enter test data")
        p = input("Enter precipitaion : ")
        t = input("Enter temperature : ")
        out_one=yieldPredict([[float(p), float(t)]])
        cropN=input("Enter N value : ")
        cropP = input("Enter P value : ")
        cropK = input("Enter K value : ")
        varietyPredict(kmeans,np.array([[cropN,cropP,cropK]]),out_one)