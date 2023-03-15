import numpy as np


def get_conv_layers(model):
    layers = [layer for layer in model.layers if 'conv' in str(layer.__class__)]
    return list(reversed(layers))

def sum_erf(layer):
    if str(layer.get_config()['kernel_initializer']).__contains__("NovelMethod"):
        # Operations if using the novel method
        rf = np.sum(layer.get_weights()[0],axis=-1,dtype=object) #Gather weight parameters matrix
        erf = np.sum(rf,axis=-1,dtype=object)         #Combine all kernels into matrix of rank 2
        return erf
    else:
        # Operations for standard dilation/initialization schemes
        dilation = layer.get_config()['dilation_rate'][0]
        kernel = layer.get_config()['kernel_size'][0]
        expanded = (kernel-(dilation-1))+(kernel*(dilation-1))
        rf = np.sum(layer.get_weights()[0],axis=-1,dtype=object)
        rf = np.sum(rf,axis=-1,dtype=object)
        erf = np.zeros([expanded,expanded])
        for x in range(rf.shape[0]):
            for y in range(rf.shape[1]):
                erf[x*dilation,y*dilation] = rf[x,y]
        return erf


def visualize_erf(layers_list):
    layers = get_conv_layers(layers_list)               #Gather list of CNN layers
    trf = np.ones([1,1])                                #Create single point TRF to start
    holding = {}                                        #Empty dictionary
    for layer in layers:                                #For each layer:
        erf = sum_erf(layer)                            #Compute layerwise ERF
        half = erf.shape[0]//2                          #Determine half size of ERF
        out_trf = np.pad(trf,half)                      #Expand the TRF based on the maximum ERF
        # print('erf',erf.shape,erf.sum())
        for x in range(trf.shape[0]):                   #For each [X,Y] position in the TRF
            for y in range(trf.shape[1]):               
                if trf[x,y] != 0:                       #If the TRF neuron is nonzero:
                    for j in range(erf.shape[0]):       #For each [J,K] position in the ERF:
                        for k in range(erf.shape[1]):
                            if erf[j,k] != 0.:          #If the ERF neuron is nonzero:
                                out_trf[j+x,k+y] += 1   #Increment the TRF at the combined location
        if trf.shape == (1,1):                          #If the TRF is a single point
            out_trf[half,half] -= 1                     #Remove the initial nonzero value from np.ones()
        holding[layer.name] = out_trf                   #Save the output TRF
        trf = out_trf                                   #Assign the output TRF for the next loop
    return holding
