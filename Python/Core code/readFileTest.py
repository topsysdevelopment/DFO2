import numpy as np
import math
from numpy import dtype

# TODO : change when the data file format will be fixed
class tableDataStructure(object):
     """Each table has a header, rowname and values"""
     def __init__(self):
         self.header=[]
         self.rowName=[]
         self.value=[]


def loadTable(file,line):
    lineList = line.split()
    if len(lineList):
        if '#' not in lineList[0] and len(lineList) == 1 :
            data={}
            while '#' not in lineList[0] and len(lineList) == 1:
                key = lineList[0]
                line = next(file)
                if len(line)>0:
                    lineList = line.split()
                    output , line = loadTable(file,line)
                    if output:
                        data[ key ] = output
                        lineList = line.split()
                        if len(lineList) == 0:
                            break
                    else:
                        break
        else:
            data = tableDataStructure()
            data.header = lineList

            line = next(file)
            lineList = line.split()
            while (len(lineList)>1):
                data.rowName.append(lineList[0])
                data.value.append(lineList[1:] )
                try:
                    line = next(file)
                except:
                    return False # catch end of the file
                lineList = line.split()
    return data , line


def readFileOfTables(path):
    ''' Works for 2 lines between each table. If dimension>2, no blank 
    lines between main name, subname1, header1,value1,subname2,... '''
    f = open(path,'r')
    data={}
    while True:
        try:
            line = next(f)
        except:
            break
        lineList = line.split()
        if len(lineList):
            if '#' not in lineList[0] and len(lineList) == 1:
                key = lineList[0]
                line = next(f)
                output = loadTable(f,line)
                if output:
                    data[ key ] = output[0]
                else:
                    break
    return data


def indexing(table, timeStepName, *args, **kwargs):
    if 'default' in kwargs:
        default_value = kwargs['default']
    else:
        default_value = 0.
        
    if len(args)==0:
        if table.header[0] in timeStepName:
            tmpTab = np.transpose(np.array(table.value,dtype='float'))
            formatTab = np.ones([len(timeStepName), len(tmpTab[0])]) * default_value 
            for i,h in enumerate(table.header):
                formatTab[ timeStepName.index(h) ] = tmpTab[i]
            return formatTab #np.transpose(np.array(table.value,dtype='float'))
        elif table.rowName[0] in timeStepName:
            tmpTab = np.array(table.value,dtype='float')
            formatTab = np.ones([len(timeStepName), len(tmpTab[0])]) * default_value 
            for i,h in enumerate(table.rowName):
                formatTab[ timeStepName.index(h) ] = tmpTab[i]
            return formatTab
        else:
            print('Table format not supported')
            raise
    elif len(args)==1:
        #find maximum dimension
        maxDim = 0
        for key in table.keys():
            maxDim = max(maxDim, len(table[key].value))
        data = np.zeros([len(timeStepName),len(table) , maxDim])
        for key in table.keys():
            tmpTab = np.array(table[key].value,dtype='float')
            formatTab = np.ones([len(tmpTab), len(timeStepName)]) * default_value 
            for i,h in enumerate(table[key].header):
                formatTab[ :, timeStepName.index(h) ] = tmpTab[:,i]
                data[:,list(args[0]).index(key),0:len(table[key].value)] = np.transpose(formatTab)
        return data
    else:
        print('Table format not supported')
        raise