import struct
import numpy as np
import os
import sys
import struct
from scipy.stats import norm, kurtosis, stats

def loadData(path_name):
    file = np.load(path_name,allow_pickle=True)
    header = None
    data = None
    if "data" in file.files :
        data = file["data"]
    elif "a" in file.files  :
        data = file["a"]
    if "header" in file.files:
        header = file["header"].item()
        #print("TRUE")
    elif "h" in file.files:
        header = file["h"] 
    return header, data  

def float_to_bin(f):
    return '0b' + format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

def bin_to_float(b):
    return struct.unpack('!f',struct.pack('!I', int(b, 2)))[0]


def compress_float(f: np.ndarray, deviation_bits=0, output="binary"):
    bitTemp = []
    if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
        return f
    binary = float_to_bin(f)
    base = binary[:0-deviation_bits] + "0" * deviation_bits
    if (output == "float"):
        return np.single(bin_to_float(base))
    return base

def compress_npndarray(f: np.single, deviation_bits=0, output="binary"):
    if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
        return f
    binary = float_to_bin(f)
    base = binary[:0-deviation_bits] + "0" * deviation_bits
    if (output == "float"):
        return np.single(bin_to_float(base))
    return base


def Compressor(f, deviation_bits=0, output="float", verbose=False, amount=10):
    print(">Compressor Starting<")
    print("Deviation used is: "+str(deviation_bits))
    length = range(0,amount)#len(f))
    #bitTemp = np.array(range(0,5))
    base = np.array(range(0,5))
    returner = list()
    #print(type(f))
    #print(f)
    if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
        return f

    for c in length:
        
        f_b_temp = float_to_bin(f[c]) #Convert to a bitwise representation
        #print(type(f))         #numpy.ndarray
        #print(type(f[c]))      #numpy.float64
        #print(type(f_b_temp))  #str
        #print(type(f_b_temp[:0-deviation_bits] + "0" * deviation_bits)) #str
        
        #bitTemp.append(float_to_bin(f[c]))
        base = f_b_temp[:0-deviation_bits] + "0" * deviation_bits   # Assemple new stored value
        cBase = f_b_temp[:0-deviation_bits]                         # What the new base looks like
        tempFlip = f_b_temp[-1:-deviation_bits:(-1)]                # Read thru bits from behind, used to gather deviation bits
        cDeviation = tempFlip[::-1]                                 # Flip found deviation bits, ready to be concatanated to cBase
        returner.append(bin_to_float(base))
        if verbose == True:
            print(">>> For-Loop begins ("+str(c)+") <<<")
            print("Float value      : "+str(f[c]))
            print("bitwiseFloat     : "+str(f_b_temp))
            print("Base             : "+str(cBase))
            print("Deviation bits   : "+str(cDeviation))
            print("Returned Value   : "+str(base))
            print("Float value post : "+str(returner[c]))
        #print(type(returner)) #'float'
        
    if (output == "float"):
        print("<< Compressor Ending >>")
        return returner #np.double(bin_to_float(base))
        # A list() with all the compressed values is returned
    print("<< Compressor Ending >>")    
    return base #type: String    

def gather_Features(Data: np.ndarray):
    #print("gather_Features")
    #N = len(Data)
    results = list()
    description = stats.describe(Data)          # 0| #samples   1| min      2| max      3| mean     4| variance 
    for ans in range(0,len(description)):       # 5| skewness   6| kurtosis 7|  RMS     8|          9|          
        if type(description[ans]) is tuple:
            results.extend(description[ans])
        else:
            results.append(description[ans])  
    results.append(np.sqrt(np.mean(Data**2))) #
    #print(description)
    #print(type(results))
    #print(type(results[1]))
    #print(description[1][0])
    #print("gather_Features!!!!!!!!")
    return np.array(results)
    #return results

def find_diff(og_info: np.ndarray, comp_info: np.ndarray):
    difference = list()
    difference.append(og_info[1] - comp_info[1])                #Min
    difference.append(og_info[2] - comp_info[2])                #Max
    difference.append(og_info[3] - comp_info[3])                #Mean
    difference.append(og_info[6] - comp_info[6])                #Kurtosis
    difference.append(og_info[7] - comp_info[7])                #RMS
    for index in range(0,len(difference)):
        if difference[index] > 0.00001:
            print(index)
    return np.array(difference)


def Compressor_Meta(f, deviation_bits=0, output="float", verbose=False, amount=10, threshhold=0.0001):
    print(">Compressor Starting< - Deviation used is: "+str(deviation_bits))
    length = range(0, amount)                                       # Used to set how much/'amount' data is compressed
    lstBase = list()                                                # List of the bases stored
    lstBaseCounter = list()                                         # Number of occurence of each base
    returner = list()                                               # Holds the compressed data
    baseCounter = 0                                                 # Used for handling storage of bases and their ID
    threshholdWarning = 0                                           # Counts the threshhold Warning

    if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
        return f

    for c in length:
        
        f_b_temp = float_to_bin(f[c])                               #Convert to a bitwise representation
        #print(type(f))                                                     #numpy.ndarray
        #print(type(f[c]))                                                  #numpy.float64
        #print(type(f_b_temp))                                              #str
        #print(type(f_b_temp[:0-deviation_bits] + "0" * deviation_bits))    #str
        
        base = f_b_temp[:0-deviation_bits] + "0" * deviation_bits   # Assemple new stored value
        cBase = f_b_temp[:0-deviation_bits]                         # What the new base looks like
        tempFlip = f_b_temp[-1:-deviation_bits:(-1)]                # Read thru bits from behind, used to gather deviation bits
        cDeviation = tempFlip[::-1]                                 # Flip found deviation bits, ready to be concatanated to cBase
        returner.append(bin_to_float(base))

        if cBase not in lstBase:                                        # Check if bases already exists
            lstBase.append(cBase)                                       # Store bases
            lstBaseCounter.append(["ID: "+str(baseCounter), 0])         # Store ID and Count of bases
            baseCounter += 1
            if verbose == True:
                print("Base not found in dictonary, adding "+str(cBase))
        else:                                                           # If base already exists
            lstBaseCounter[lstBase.index(cBase)][1] += 1                # Incremnt counter for that ID
            if verbose == True:
                print("Already found as base number "+str(lstBase.index(cBase)))

        if((f[c] - returner[c])> threshhold):
            threshholdWarning += 1

        if verbose == True:
            print(">>> For-Loop begins ("+str(c)+") <<<")
            print("Float value      : "+str(f[c]))
            print("bitwiseFloat     : "+str(f_b_temp))
            print("Base             : "+str(cBase))
            print("Deviation bits   : "+str(cDeviation))
            print("Returned Value   : "+str(base))
            print("Float value post : "+str(returner[c]))
            if((f[c] - returner[c])> threshhold):                         # If difference in value above 'threshhold' let us know
                print("Difference in Loop "+str(c)+" Value after Compression bigger than : "+str(threshhold))
            
        #print(type(returner)) #'float'
    if verbose == True:  
        print("Number of bases found : "+str(len(lstBase))+" using "+str(len(f))+" Samples and "+str(deviation_bits)+" deviation bits")
        print("Bases stored : "+str(lstBase))
        print("Base ID's and counts: "+str(lstBaseCounter))
        print("Number of threshhold warnings : "+str(threshholdWarning)+" using "+str(len(f))+" Samples")

    if (output == "float"):
        #print("<< Compressor Ending >>")
        return returner #np.double(bin_to_float(base))
        # A list() with all the compressed values is returned
    print("<< Compressor Ending >>")    
    return base #type: String    

# Attempt to add base storage and checking
def Compressor(f, deviation_bits=0, output="float", verbose=False, amount=10, threshhold=0.0001):
    print(">Compressor Starting<")
    print("Deviation used is: "+str(deviation_bits))
    length = range(0, amount)#len(f))
    lstBase = list()
    lstBaseCounter = list()
    returner = list()
    baseCounter = 0
    threshholdCounter = 0
    #print(type(f))
    #print(f)
    if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
        return f

    for c in length:
        
        f_b_temp = float_to_bin(f[c]) #Convert to a bitwise representation
        #print(type(f))                                                     #numpy.ndarray
        #print(type(f[c]))                                                  #numpy.float64
        #print(type(f_b_temp))                                              #str
        #print(type(f_b_temp[:0-deviation_bits] + "0" * deviation_bits))    #str
        
        base = f_b_temp[:0-deviation_bits] + "0" * deviation_bits   # Assemple new stored value
        cBase = f_b_temp[:0-deviation_bits]                         # What the new base looks like
        tempFlip = f_b_temp[-1:-deviation_bits:(-1)]                # Read thru bits from behind, used to gather deviation bits
        cDeviation = tempFlip[::-1]                                 # Flip found deviation bits, ready to be concatanated to cBase
        returner.append(bin_to_float(base))

        if cBase not in lstBase:                                        # Check if bases already exists
            lstBase.append(cBase)                                       # Store bases
            lstBaseCounter.append(["ID: "+str(baseCounter), 0])            # Store ID and Count of bases
            baseCounter += 1
            if verbose == True:
                print("Base not found in dictonary, adding "+str(cBase))
        else:                                                           # If base already exists
            lstBaseCounter[lstBase.index(cBase)][1] += 1                # Incremnt counter for that ID
            if verbose == True:
                print("Already found as base number "+str(lstBase.index(cBase)))

        if((f[c] - returner[c])> threshhold):
            threshholdCounter += 1

        if verbose == True:
            print(">>> For-Loop begins ("+str(c)+") <<<")
            print("Float value      : "+str(f[c]))
            print("bitwiseFloat     : "+str(f_b_temp))
            print("Base             : "+str(cBase))
            print("Deviation bits   : "+str(cDeviation))
            print("Returned Value   : "+str(base))
            print("Float value post : "+str(returner[c]))
            if((f[c] - returner[c])> threshhold):                         # If difference in value above 'threshhold' let us know
                print("Difference in Loop "+str(c)+" Value after Compression bigger than : "+str(threshhold))
            
        #print(type(returner)) #'float'
    if verbose == True:  
        print("Number of bases found : "+str(len(lstBase))+" using "+str(len(f))+" Samples and "+str(deviation_bits)+" deviation bits")
        print("Bases stored : "+str(lstBase))
        print("Base ID's and counts: "+str(lstBaseCounter))
        print("Number of threshhold warnings : "+str(threshholdCounter)+" using "+str(len(f))+" Samples")

 
    return returner, lstBase, lstBaseCounter #type: String    


def Compressor_Meta(f, deviation_bits=0, output="float", verbose=False, amount=10, threshhold=0.0001):
    print(">Compressor Starting<")
    print("Deviation used is: "+str(deviation_bits))
    length = range(0, amount)#len(f))
    lstBase = list()
    lstBaseCounter = list()
    returner = list()
    baseCounter = 0
    threshholdCounter = 0
    #print(type(f))
    #print(f)
    #if (deviation_bits == 0): #If no deviation is used, we return pure data - The transformation only results in a "base"
    #    return f

    for c in length:
        
        f_b_temp = float_to_bin(f[c]) #Convert to a bitwise representation
        #print(type(f))                                                     #numpy.ndarray
        #print(type(f[c]))                                                  #numpy.float64
        #print(type(f_b_temp))                                              #str
        #print(type(f_b_temp[:0-deviation_bits] + "0" * deviation_bits))    #str
        
        base = f_b_temp[:0-deviation_bits] + "0" * deviation_bits   # Assemple new stored value
        cBase = f_b_temp[:0-deviation_bits]                         # What the new base looks like
        tempFlip = f_b_temp[-1:-deviation_bits:(-1)]                # Read thru bits from behind, used to gather deviation bits
        cDeviation = tempFlip[::-1]                                 # Flip found deviation bits, ready to be concatanated to cBase
        returner.append(bin_to_float(base))

        if cBase not in lstBase:                                        # Check if bases already exists
            lstBase.append(cBase)                                       # Store bases
            lstBaseCounter.append(["ID: "+str(baseCounter), 0])            # Store ID and Count of bases
            baseCounter += 1
            if (verbose == True):
                print("Base not found in dictonary, adding "+str(cBase))
        else:                                                           # If base already exists
            lstBaseCounter[lstBase.index(cBase)][1] += 1                # Incremnt counter for that ID
            if (verbose == True):
                print("Already found as base number "+str(lstBase.index(cBase)))

        if((f[c] - returner[c])> threshhold):                           # If difference in value above 'threshhold' let us know
            threshholdCounter += 1

        if (verbose == True) and (amount < 10):
            print(">>> For-Loop begins ("+str(c)+") <<<")
            print("Float value      : "+str(f[c]))
            print("bitwiseFloat     : "+str(f_b_temp))
            print("Base             : "+str(cBase))
            print("Deviation bits   : "+str(cDeviation))
            print("Returned Value   : "+str(base))
            print("Float value post : "+str(returner[c]))

    if verbose == True:  
        print("Number of bases found : ("+str(len(lstBase))+") using "+str(amount)+" Samples and "+str(deviation_bits)+" deviation bits")
        print("Bases stored : "+str(lstBase))
        print("Base ID's and counts: "+str(lstBaseCounter))
        print("Number of threshhold warnings : "+str(threshholdCounter)+" using "+str(amount)+" Samples")
        print("type of lstbase"+str(type(lstBase)))
        print("type of lstbaseCOUNTER"+str(type(lstBaseCounter[0])))
        print("type of lstbaseCOUNTER"+str(type(lstBaseCounter)))

    return returner, lstBase, lstBaseCounter


def Compressor_Meta_lists(f):                              # Compress data and return 0-32 devations
    results_Data        = list()
    # results_Deviations  = list()
    results_Bases       = list()
    results_Counts      = list()
    temp_data           = list()
    temp_base           = list()
    temp_counts         = list()
    for idx in range(0,33):
        temp_data.clear(); temp_base.clear(); temp_counts.clear(); baseCounter = 0
        print("LOOP :"+str(idx))

        for c in range(0,len(f)):
            binary = float_to_bin(f[c])
            if idx == 0:
                base = binary[:-1-idx] + "0"                          # if deviation bits 0 dont multiply by 0
            else:
                base = binary[:0-idx] + "0" * idx                   # Assemple new stored value
            #tempFlip = binary[-1:-index-1:(-1)]                         # Read thru bits from behind, used to gather deviation bits
            #cDeviation = tempFlip[::-1]                                 # Flip found deviation bits, ready to be concatanated to cBase
            temp_data.append(np.float64(bin_to_float(base)))
            if base not in temp_base:
                temp_base.append(base)
                temp_counts.append(["ID: "+str(baseCounter), 1, "deviations bits : "+str(idx)])
                baseCounter += 1
            else:
                temp_counts[temp_base.index(base)][1] += 1
            if c == len(f)-1:
                results_Data.append(temp_data[:])
                results_Bases.append(temp_base[:])
                results_Counts.append(temp_counts[:])
                print("SAVE")

    return results_Data, results_Bases, results_Counts


def compression_Ratios(DEVIATION_BITS, ORIGINAL_DATA, verbose = False):
    setter = 0; totalSamples = 0; highest = 0; IDbits = 0; Countsbits = 0
    sizeOG = len(ORIGINAL_DATA) * 32             #16 Million bits
    dataBits = lstBase[DEVIATION_BITS][setter][2:(34-DEVIATION_BITS):]                            #Data bits
    compressionBits = lstBase[DEVIATION_BITS][setter][-1:(33-DEVIATION_BITS):-1] 

    for IDcounter in range(0,28):
        IDahead = IDcounter+1
        if (((len(lstBase[DEVIATION_BITS])) > np.power(2,IDcounter)) & ((len(lstBase[DEVIATION_BITS])) < np.power(2,IDahead))):
            IDbits = IDcounter+1
            break
        
    for TScounter in range(0, len(lstBase[DEVIATION_BITS])):
        current = lstCounts[DEVIATION_BITS][TScounter][1].astype(int)
        totalSamples += lstCounts[DEVIATION_BITS][TScounter][1].astype(int) 
        if current > highest:
            highest = current

    for ccounter in range(0,28):
        cahead = ccounter+1
        if ((highest > np.power(2,ccounter)) and (highest < np.power(2,cahead))):
            Countsbits = ccounter+1
            break
            
    #                  Amount of data bits * number of bases
    compressedBase      = ((len(dataBits)) * len(lstBase[DEVIATION_BITS]))
    #                  Bits needed to store counts * number of bases
    compressedCounts    = Countsbits * len(lstBase[DEVIATION_BITS])
    #                  Amount of devi bits * number of samples
    compressedDevi      = DEVIATION_BITS * len(d_og)
    # IDs = bits need to store numbers = number of bases
    compressedID = IDbits * len(d_og)


    #metric 1 base + count
    metric1 = compressedBase + (compressedCounts)
    metric1Ratio = metric1 / sizeOG
    #metric 2 base + ids
    metric2 = compressedBase + compressedID
    metric2Ratio = metric2 / sizeOG
    #metric 3 base + id + devication
    metric3 = compressedBase + compressedID + compressedDevi
    metric3Ratio = metric3 / sizeOG

    if (verbose == True):
        print(">>Compression ratio finder<<")
        print("Size of the original data       : "+str(sizeOG)+" bits")
        print("Data bits >"+str(len(lstBase[DEVIATION_BITS][setter])-2-DEVIATION_BITS)+"< & Deviation bits >"+str(DEVIATION_BITS)+"< & Number of bases >"+str(len(lstBase[DEVIATION_BITS]))+"<")
        print("Random look at data point")
        print(dataBits+"|"+compressionBits)
        print("IDs need     "+str(IDbits)+" bits")
        print("Counts need  "+str(Countsbits)+" bits")    
        print("Highest value of count     : "+str(highest))

        print("Bits needed to store Bases       : "+str(compressedBase))
        print("Bits needed to store Counts      : "+str(compressedCounts))
        print("Bits needed to store Deviations  : "+str(compressedDevi))
        print("Bits needed to store ID's        : "+str(compressedID))

        print("Metric 1 - Base + Count          : "+str(metric1Ratio*100)+"%")
        print("Metric 2 - Base + ID             : "+str(metric2Ratio*100)+"%")
        print("Metric 3 - Base + ID + Devication: "+str(metric3Ratio*100)+"%")

        
    return metric1Ratio*100, metric2Ratio*100, metric3Ratio*100, metric1, metric2, metric3, sizeOG, DEVIATION_BITS
   