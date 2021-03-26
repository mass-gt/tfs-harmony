# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:17:13 2020

@author: modelpc
"""
import pandas as pd
import numpy as np
import shapefile as shp
import array
import os.path


def read_mtx(mtxfile):  
    '''
    Read a binary mtx-file (skimTijd and skimAfstand)
    '''
    mtxData = array.array('i')  # i for integer
    mtxData.fromfile(open(mtxfile, 'rb'), os.path.getsize(mtxfile) // mtxData.itemsize)
    
    # The number of zones is in the first byte
    mtxData = np.array(mtxData, dtype=int)[1:]
    
    return mtxData


def write_mtx(filename, mat, aantalZones):
    '''
    Write an array into a binary file
    '''
    mat     = np.append(aantalZones,mat)
    matBin  = array.array('i')
    matBin.fromlist(list(mat))
    matBin.tofile(open(filename, 'wb'))


def read_shape(shapePath, encoding='latin1', returnGeometry=False):
    '''
    Read the shapefile with zones (using pyshp --> import shapefile as shp)
    '''
    # Load the shape
    sf = shp.Reader(shapePath, encoding=encoding)
    records = sf.records()
    if returnGeometry:
        geometry = sf.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf.fields
    sf.close()
    
    # Get information on the fields in the DBF
    columns  = [x[0] for x in fields[1:]]
    colTypes = [x[1:] for x in fields[1:]]
    nRecords = len(records)
    
    # Put all the data records into a NumPy array (much faster than Pandas DataFrame)
    shape = np.zeros((nRecords,len(columns)), dtype=object)
    for i in range(nRecords):
        shape[i,:] = records[i][0:]
    
    # Then put this into a Pandas DataFrame with the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if colTypes[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if colTypes[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)
    
    if returnGeometry:
        return (shape, geometry)
    else:
        return shape
    