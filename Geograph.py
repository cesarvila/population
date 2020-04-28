
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import shapefile as shp

def plot_shape(id, sf, s=None):
    plt.figure()
    #plotting the graphical axes where map ploting will be done
    ax = plt.axes()
    ax.set_aspect('equal')#storing the id number to be worked upon
    shape_ex = sf.shape(id)#NP.ZERO initializes an array of rows and column with 0 in place of each elements
    #an array will be generated where number of rows will be(len(shape_ex,point))and number of columns will be 1 and stored into the variable
    x_lon = np.zeros((len(shape_ex.points),1))#an array will be generated where number of rows will be(len(shape_ex,point))and number of columns will be 1 and stored into the variable
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]#plotting using the derived coordinated stored in array created by numpy
    plt.plot(x_lon,y_lat)
    x0 = np.mean(x_lon)
    y0 = np.mean(y_lat)
    plt.text(x0, y0, s, fontsize=10)# use bbox (bounding box) to set plot limits
    plt.xlim(shape_ex.bbox[0],shape_ex.bbox[2])
    return x0, y0

def plot_map(sf, x_lim = None, y_lim = None, figsize = (11,9)):
    plt.figure(figsize = figsize)
    id=0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')

        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id+1

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

#calling the function and passing required parameters to plot the full map

def plot_map_fill(id, sf, x_lim = None,
                          y_lim = None,
                          figsize = (11,9),
                          color = 'r'):

    plt.figure(figsize = figsize)
    fig, ax = plt.subplots(figsize = figsize)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    shape_ex = sf.shape(id)
    x_lon = np.zeros((len(shape_ex.points),1))
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    ax.fill(x_lon,y_lat, color)

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        #plot_map_fill(0, sf, x_lim, y_lim, color=’y’)plot_map_fill(13, sf,color=’y’)
def plot_map_fill_multiples_ids(title, country, sf,
                                               x_lim = None,
                                               y_lim = None,
                                               figsize = (11,9),
                                               color = 'r'):


    plt.figure(figsize = figsize)
    fig, ax = plt.subplots(figsize = figsize)
    fig.suptitle(title, fontsize=16)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    for id in country:
        shape_ex = sf.shape(id)
        x_lon = np.zeros((len(shape_ex.points),1))
        y_lat = np.zeros((len(shape_ex.points),1))
        for ip in range(len(shape_ex.points)):
            x_lon[ip] = shape_ex.points[ip][0]
            y_lat[ip] = shape_ex.points[ip][1]
        ax.fill(x_lon,y_lat, color)

        x0 = np.mean(x_lon)
        y0 = np.mean(y_lat)
        plt.text(x0, y0, id, fontsize=10)

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

def plot_countries_2(sf, title, countries, color):

    df = read_shapefile(sf)
    country_id = []
    for i in countries:
        country_id.append(df.index[df['COUNTRY'] == i.title()][0])
    plot_map_fill_multiples_ids(title, country_id, sf,
                                       x_lim = None,
                                       y_lim = None,
                                       figsize = (11,9),
                                       color = color);

def calc_color(data, color=None):
    if color   == 1:
        color_sq =  ['#dadaebFF','#bcbddcF0','#9e9ac8F0','#807dbaF0','#6a51a3F0','#54278fF0'];
        colors = 'Purples';
    elif color == 2:
        color_sq = ['#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494'];
        colors = 'YlGnBu';
    elif color == 3:
        color_sq = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#636363','#252525'];
        colors = 'Greys';
    elif color == 9:
        color_sq = ['#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000'];

    else:
        color_sq = ['#ffffd4','#fee391','#fec44f','#fe9929','#d95f0e','#993404'];
        colors = 'YlOrBr';
    new_data, bins = pd.qcut(data, 6, retbins=True,
    labels=list(range(6)))
    color_ton = []
    for val in new_data:
        color_ton.append(color_sq[val])
    if color != 9:
        colors = sns.color_palette(colors, n_colors=6)
        sns.palplot(colors, 0.6);
        for i in range(6):
            print ("\n"+str(i+1)+': '+str(int(bins[i]))+
                   " => "+str(int(bins[i+1])-1))
        print("\n\n   1   2   3   4   5   6")
    return color_ton, bins;

def plot_countries_data(sf, title, countries, data=None,color=None, print_id=False):

    color_ton, bins = calc_color(data, color)
    df = read_shapefile(sf)
    country_id = []
    for i in countries:
        country_id.append(df.index[df['COUNTRY'] == i.title()][0])
    plot_map_fill_multiples_ids_tone(sf, title, country_id,
                                     print_id,
                                     color_ton,
                                     bins,
                                     x_lim = None,
                                     y_lim = None,
                                     figsize = (11,9));

def plot_map_fill_multiples_ids_tone(sf, title, country,
                                     print_id, color_ton,
                                     bins,
                                     x_lim = None,
                                     y_lim = None,
                                     figsize = (11,9)):


    plt.figure(figsize = figsize)
    fig, ax = plt.subplots(figsize = figsize)
    fig.suptitle(title, fontsize=16)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    for id in country:
        shape_ex = sf.shape(id)
        x_lon = np.zeros((len(shape_ex.points),1))
        y_lat = np.zeros((len(shape_ex.points),1))
        for ip in range(len(shape_ex.points)):
            x_lon[ip] = shape_ex.points[ip][0]
            y_lat[ip] = shape_ex.points[ip][1]
        ax.fill(x_lon,y_lat, color_ton[country.index(id)])
        if print_id != False:
            x0 = np.mean(x_lon)
            y0 = np.mean(y_lat)
            plt.text(x0, y0, id, fontsize=10)
    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)
