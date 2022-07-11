import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as scio
from scipy import signal
import os
#import pdb
import pandas as pd #maybe remove this dependency eventually
from grid_strategy import strategies
import warnings


class DataLoader():
    '''
    Takes a mat-file name as input and tries to read the data from it using either h5py or scipy.io (depends on version of matfile)
    exposes the "data" matrix, "experParams" struct and "samplingRate"
    '''
    def  __init__(self, fname):
        try:
            with h5py.File(fname, 'r') as hf:
                self.data = hf['data'][:]
                try:
                    self.samplingRate = hf['experParams']['srate'][0,0]
                    self.experParams = dict(hf['experParams'])
                except KeyError:#when I saved using hdf5storage, the experParams is not included, didn't work
                    self.samplingRate = hf['samplingRate'].value[0,0]#that's what it's called in that case
        except OSError:
            self.matfile = scio.loadmat(fname)
            self.data = self.matfile['data']
            self.samplingRate = self.matfile['experParams'][0,0][4][0,0]
        self.samplingRate = round(self.samplingRate/100.)*100.
        self.file=fname
        if self.data.shape[0]>self.data.shape[1]:#assert the shapes are as assumed later
            self.data = self.data.T
        #AS stands for arena signal
        self.ASchannel = -2
        self.metadata = dict()
        self.extract_metadata()
        return
        
    def get_data(self):
        return self.data, self.samplingRate
    
    def extract_metadata(self):
        filename = os.path.basename(self.file)
        filename = filename.rstrip('.mat')
        metas = filename.split('_')
        i=0
        yearRange = [str(y) for y in range(2010,2025)]
        while i<len(metas):
            if not metas[i]:
                i+=1
                continue
            if metas[i][:3].lower()=='fly':
                self.metadata['fly'] = int(metas[i][3:])
                i+=1
            elif 'x' in metas[i] or metas[i][:2].upper() == 'WT' or ( metas[i][0].upper() in ('G','U','X','B','L','J','R') and metas[i][1:3].isnumeric() ) or metas[i].lower()=='na':
                self.metadata['cross'] = metas[i]
                i+=1
            elif metas[i] =='age':
                if metas[i+1].lower()=='na':
                    self.metadata['age'] = -1
                else:
                    assert(metas[i+1][-1]=='d')
                    self.metadata['age'] = int(metas[i+1][:-1])
                i+=2
            elif 'cell' == metas[i][:4]:
                try:
                    self.metadata['cell'] = int(metas[i].lstrip('cell'))
                except ValueError:
                    self.metadata['cell'] = metas[i].lstrip('cell')
                i+=1
            elif metas[i] in ['looming','direction','optogeneticsTests']:
                if metas[i] == 'direction':
                    self.metadata['stimprotocol'] = metas[i] + '_' + metas[i+1]
                    i+=2
                elif metas[i] == 'looming':
                    if metas[i+1] == 'wb':
                        self.metadata['stimprotocol'] = '_'.join(metas[i:i+2])
                        i+=2
                    else:
                        self.metadata['stimprotocol'] = metas[i]
                        i+=1
                elif metas[i] == 'optogeneticsTests':
                    self.metadata['stimprotocol'] = metas[i] + '_' + metas[i+1]
                    i+=2
                else:
                    print('Skipped stimulus protocol metadata, not fully implemented yet')
                    i+=1
            #elif metas[i] in ['2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028','2029','2030']:
            elif metas[i] in yearRange:
                self.metadata['year'] = int(metas[i])
                self.metadata['month'] = int(metas[i+1])
                self.metadata['day'] = int(metas[i+2])
                self.metadata['hour'] = int(metas[i+3])
                self.metadata['minute'] = int(metas[i+4])
                self.metadata['second'] = int(metas[i+5])
                i+=6
            else:
                #placeholder for other stuff without always outputting a lot of garbage via print or warning
                self.metadata[metas[i]]=''
                i+=1
        return

    def get_metadata(self):
        return self.metadata
    
    def get_Nchannels(self):
        return self.data.shape[0]
    
    def plot(self):
        ''' plots the raw data as a whole with the correct timeAx '''
        self.f_rawdataAll , self.ax_rawdataAll = plt.subplots(1)
        self.timeAx = np.arange(max(self.data.shape))/self.samplingRate
        self.ax_rawdataAll.plot(self.timeAx, self.data[:-2,:].T)
        self.ax_rawdataAll.plot(self.timeAx, self.data[-2:,:].T, alpha=0.3)
        return
    
    def get_fig_ax(self):
        ''' return the fiure and axis objects of the rawdata plot '''
        return self.f_rawdataAll, self.ax_rawdataAll
        
    def get_timeAx(self):
        return self.timeAx
        
    def get_stimulusBoundaries(self):
        stimIDsignal = self.data[-1,:]#last channel is the signal given by the program to distinguish stimuli
        thresh = 0.05 #it's very clean
        stimIDcrossings = np.where(np.diff(stimIDsignal>thresh))[0] #no distinction between begin/end
        return stimIDcrossings






'''
## segment raw data for averaging
#find alll signal areas and their value
stimSignal = data[-1,:] #last channel is the signal given by the program to distinguish stimuli
thresh = 0.05 #it's very clean
stimIDcrossings = np.where(np.diff(stimSignal>thresh))[0] #no distinction between begin/end
#confirm by plotting
xval = ax_rawdataAll.get_children()[0].get_xdata()
ax_rawdataAll.plot(xval[crossings], stimSignal[crossings], 'ko')
#ok, then determine the value of that segment with and group them together
segValue = [np.mean(stimSignal[crossings[i]+1:crossings[i+1]]) for i in range(0,crossings.size,2)]
'''


class Segment():
    def __init__(self, boundaries, meanValue, datasegment, **kwargs):
        self.boundaries=boundaries
        self.meanValue = meanValue
        self.data = datasegment
        self.samplingRate = kwargs.get('Fs',None)
        self.numel = self.data.shape[1]
        #self.channelmap = kwargs.get('channelmap', {'wingL':0,'wingR':1,'arenaOut':2,'stimID':3})    
        
    def get_signal(self):
        return self.meanValue
    def __len__(self):
        return self.data.shape[1]
    

class SegmentCollection():
    '''
    This class is meant to keep a collection of Segments, which are individually defined to hold the piece of data that is indentified by a stimulus signalling pulse
    Instead of using it directly, in most cases it should be used as a base class to build an Analyzer class adapted to the specific experiment protocols (for example see class below)'''
    def __init__(self, crossings, data, **kwargs):
        if data.shape[0]>data.shape[1]:#data wrong format
            data = data.T
        stimSignal = data[-1,:]
        segValue = [np.mean(stimSignal[crossings[i]+10:crossings[i+1]]) for i in range(0,crossings.size,2)]
        self.segments = np.array([ Segment(boundaries=crossings[i:i+2], meanValue=segValue[i//2],
                                  datasegment=data[:,crossings[i]:crossings[i+1]]) for i in range(0,crossings.size,2) ])
        self.samplingRate = kwargs.get('Fs',None)
        #possibility for how many data channels there are
        #if 'nChannels' in kwargs.keys():
        #    self.nChannels = kwargs.nChannels
        #else:
        #    self.nChannels = data.shape[0]-2 #subtract 2 to only have the wing and ephys channels, simpler to imagine
        self.nChannels = kwargs.get('nChannels',data.shape[0]-2)
        self.keepTrackofCollections = np.zeros(len(self.segments))
        self.timeax = kwargs.get('timeax',np.arange(max(data.shape)))
        if self.samplingRate is not None:
            self.timeax = self.timeax/self.samplingRate
        self.group_by_signal() #defines the groups and sorts accordingly
        return
        
    
    def group_by_signal(self):
        #using np.histogram or manually
        tolerance = 0.1*2
        allValues = pd.Series([seg.get_signal() for seg in self.segments])#pandas makes handling nans much nicer
        recipient = pd.Series([None]*len(allValues))
        i=0
        while not all(np.isnan(allValues)):
            start = allValues.min()
            selector = allValues.between(start-0.05, start+tolerance, inclusive=True)#inclusive=True is default
            recipient[selector] = i
            allValues[selector] = np.nan
            if i>allValues.size: #emergency brake
                Exception('Count too high, in SegmentCollection')
            i+=1
        self.group_indices = recipient.to_numpy(dtype='int') #conversion important, else indexing can lead to weird results later on
        self.group_indices_sortlist = recipient.index
        self.sort_by_signal()
        return
    
    def sort_by_signal(self):
        '''even though it's not necessary at first, 
        better do it because problems have arisen later at the plotting stage, 
        confusing the interpretation, when later you don't really pay attention 
        anymore to the fact that they're not sorted while iterating
        Actually it was fine all along, because it remained consistent with subplot indices, 
        but this is more prudent'''
        sortIdx = self.group_indices.argsort()
        self.segments = self.segments[sortIdx]
        self.group_indices = self.group_indices[sortIdx]
        self.group_indices_sortlist = self.group_indices_sortlist[sortIdx]
        self.originalOrder = sortIdx
        self.keepTrackofCollections = self.keepTrackofCollections[sortIdx]
        #make sure it's correct now
        assert(not any(np.diff(self.group_indices)<0) )
        
    
    def __len__(self):
        return self.segments.shape[-1]
    
    def get_by_group(self, idx):
        if not isinstance(idx, (int, np.integer)):
            raise TypeError('index needs to be int')
        if idx <= self.group_indices.max():
            temp = self.segments[self.group_indices==idx]
            recipient = np.nan*np.ones(( len(temp), temp[0].data.shape[0], min([len(t) for t in temp]) ))
            for i in range(len(temp)):
                #pad datasegments if necessary
                #t = np.pad(temp[i].data, ((0,0),(0,recipient.shape[-1]-temp[i].data.shape[-1])), 'constant',constant_values=np.nan )
                recipient[i,:,:] = temp[i].data[:, :recipient.shape[2]]
            return recipient
        return None
    
    def average_by_stimSignal(self):
        self.meanSegments = {}
        self.stdSegments = {}
        for cat in set(self.group_indices):
            temp=self.get_by_group(cat)
            #self.meanSegments = self.meanSegments.join(pd.DataFrame(np.nanmean(temp, axis=0)))
            self.meanSegments[cat] = np.nanmean(temp, axis=0)
            self.stdSegments[cat] = np.nanstd(temp, axis=0)

    def plotMean(self):
        try:
            self.meanSegments
        except AttributeError:
            self.average_by_stimSignal()
        specs = strategies.RectangularStrategy().get_grid(self.group_indices.max()-self.group_indices.min()+1)
        f=plt.gcf()
        ax=[f.add_subplot(sp) for sp in specs]
        #f,ax = plt.subplots(2,4)
        #ax = ax.flatten()
        for k in self.meanSegments.keys():
            for i in range(self.meanSegments[k].shape[0]):
                ax[k].plot(self.timeax[:max(self.meanSegments[k].shape)], self.meanSegments[k][i,:])
                ax[k].fill_between(self.timeax[:max(self.meanSegments[k].shape)], self.meanSegments[k][i]+self.stdSegments[k][i] , self.meanSegments[k][i]-self.stdSegments[k][i], alpha=0.5)

    def plot_by_stimSignal(self):
        try:
            self.meanSegments
        except AttributeError:
            self.average_by_stimSignal()
        #plt.figure()
        nCols = len(set(self.group_indices)) #is different from self.group_indices.max() when all instances of one group get removed or merged to other fly
        fig, ax = plt.subplots(self.nChannels, nCols)
        for i in set(self.group_indices):
            plotIdx = (i//4)*4+i+1
            temp=self.get_by_group(i)
            for t in temp:
                if t.ndim==1:
                    t=t.reshape(1,t.size)
                for k in range(self.nChannels):
                    #ax[plotIdx+k*nCols].plot(t[k,:])
                    ax[k].plot(t[k,:])
            for k in range(self.nChannels):
                #ax[plotIdx+k*nCols].plot( self.meanSegments[i][k,:], color='k' )
                ax[k].plot( self.meanSegments[i][k,:], color='k' )
            #for t in temp:
                #plt.subplot(4,4,plotIdx)
                #plot wingbeat L and R
                #plt.plot(t[0,:])
                #plt.subplot(4,4,plotIdx+4)
                #plt.plot(t[1,:])
            #plt.subplot(4,4,plotIdx)
            #plt.plot(self.meanSegments[i][0,:], color='k')
            #plt.subplot(4,4,plotIdx+4)
            #plt.plot(self.meanSegments[i][1,:], color='k')
        #plt.subplot(4,4,1)
        #plt.ylabel('wingbeat L')
        #plt.subplot(4,4,5)
        #plt.ylabel('wingbeat R')
        for k in range(self.nChannels):
            #ax[plotIdx+k*nCols].set_ylabel(channelLabels[k])
            ax[k].set_ylabel(self.channelLabels[k])
        return fig
        
    def add_segment(self, newSegment):
        self.segments = np.append(self.segments, newSegment)
        self.group_by_signal()
        self.average_by_stimSignal()
            
    def add_collection(self, newCollection):
        self.__add__(newCollection)
        
    def __add__(self, newCollection):
        #assert(type(newCollection)==type(self))
        self.segments = np.concatenate((self.segments, newCollection.segments))
        #self.segments = np.c_[self.segments, newCollection]
        newCollTrackNr = (np.max(self.keepTrackofCollections)+1) *np.ones(newCollection.keepTrackofCollections.shape) + newCollection.keepTrackofCollections #add old in case it's made of more than one original collection
        self.keepTrackofCollections = np.concatenate( (self.keepTrackofCollections, newCollTrackNr) )
        self.group_by_signal()
        SegmentCollection.average_by_stimSignal(self)
        return self
        
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def get_averaged_segment(self,idx):
        try:
            self.meanSegments
        except AttributeError:
            self.average_by_stimSignal()
        return self.meanSegments[idx]

    def get_by_channel(self, idx):
        if not isinstance(idx, (int, np.integer)):
            raise TypeError('index needs to be int')
        if idx < self.segments[0].data.shape[0]:
            #maxLen = max([len(t) for t in self.segments])
            minLen = min([len(t) for t in self.segments])
            recipient = np.nan*np.ones((len(self.segments), minLen))
            for i,seg in enumerate(self.segments):
                #recipient[i,:] = np.pad(seg.data[idx], (0, maxLen-seg.data.shape[1]), 'constant',constant_values=np.nan )
                recipient[i,:] = seg.data[idx,:minLen]
            return recipient
        else:
            raise IndexError


            

class BaseAnalyzer(SegmentCollection):
    '''
    version 1.2.0 includes neo IO functions. writing Nix files technically working, but unknown issue bloats the files, neomat is fine
    version 1.2.1 plot_stimPhases is now a virtual function
    version 1.2.4 replace numpy dtype declarations e.g. np.float -> float because deprecation warning
    version 1.2.5 modifies __add__ and retrieveFromNeomatfile to account for empty or single-segment Collections
    version 1.2.6 adds np.number to type checks when replacing dict keys for neomat-files
    '''
    stimulusPhases_inframes = [] # no prior information about stimulus
    BaseAnalyzer_version = '1.2.5'#to keep track of class versions, makes easier to check for changes. Starts with 1.1.1 on 18.1.2020
    preStim=0.
    
    def __init__(self, *args, **kwargs):
        #super(BaseAnalyzer, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        if self.nChannels == 2:
            self.channelLabels = ['wingL','wingR']
        else:
            self.channelLabels = ['U [V]', 'wingL','wingR']
        self.create_channelmap()
        self.qualityTheta = kwargs.get('qualityTheta', 0.15)
        self.make_wingDiffs()
        if not 'meanFramePeriod' in kwargs:
            self.frame_update_periods = self.get_frame_periods()
            self.meanFramePeriod = np.median(self.frame_update_periods)
        else:
            self.meanFramePeriod = kwargs['meanFramePeriod']
        self.fileOrigin = {0: kwargs.get('fname', 'NA')}
        self.metadata = {0:{}}
        self.flyID = {}
        if 'metadata' in kwargs.keys():
            self.set_metadata(kwargs['metadata'], 0) #also sets flyID
            #self.flyID = {0: 'fly'+str(self.metadata['fly'])+'_'+str(self.metadata['year'])+str(self.metadata['month'])+str(self.metadata['day']) }#this is easier to compare than the dict
        self.fly_indices = np.zeros(self.segments.size, dtype=np.int8)
        self.flying = np.ones(len(self.segments),dtype=np.bool)
        self._adjust_flyIndices_order()
        #self.removedSegs = np.zeros(0, dtype='O')
        #self.remove_duds()
        #   self.meanWingDiff_per_stim_fly = {}
               
    def get_frame_periods(self):
        return self.get_frame_periods__NEW()
        
    def get_frame_periods__NEW(self):
        '''return the refresh period of the arena screen, not Kinefly
         that result is used to e.g. make the plot pretty
         This is a new method, works better than the old one (first method is deleted),
         hardenend against noise'''
        #ao = seg.data[self.channelmap['arenaOut'], :int(30*loader.samplingRate)]
        #res = np.zeros(len(self.segments)//2)
        res = [None]*len(self.segments)
        for j,seg in enumerate(self.segments):
            ao = seg.data[self.channelmap['arenaOut'], :]
            fao=ao
            w = int(0.0015*self.samplingRate+0.5)
            if w%2==0: w+=1 #making sure it's an odd number
            for i in range(3):
                fao = signal.medfilt(fao,w) #repeated median filtering
            dfao = np.diff(fao)
            dfao = signal.medfilt(dfao, 7)
            winWidth = 26
            dfao = np.convolve(dfao, signal.get_window('hamming',winWidth), 'full')[winWidth//2:]
            zdfao = (dfao-dfao.mean())/dfao.std()
            try: #allow for possibility to give some rough minimum frame rate [s] in derived classes
                peakSuppressionWindow = self.minFramePeriod*self.samplingRate
            except AttributeError:
                peakSuppressionWindow = 0.006*self.samplingRate #be conservative with hardcoded values
            peaks = signal.find_peaks(np.abs(zdfao), height=2.0, distance=peakSuppressionWindow)[0]+1 #absolute values are important here
            #additional refinement to remove some out of stimulus ones (if any)
            stimID = seg.data[self.channelmap['stimID'], :]
            inStim = np.where(np.diff(stimID>0.4))[0]
            if len(inStim)!=2:
                #raise Warning('Corrupt data, contains more/less than one stimulus!')
                warnings.warn('Corrupt data supected, contains more/less than one stimulus!', category=UserWarning)
            peaks = peaks[peaks>inStim[0]-5]
            peaks = peaks[peaks<inStim[1]+5 ]
            #get periods
            dpeaks = np.diff(peaks)
            if np.std(dpeaks) >0.005*self.samplingRate: #somethings off, this is a considerable std
                dpeaks = dpeaks[ dpeaks<np.median(dpeaks)+3.*np.std(dpeaks)]
                #dpeaks = dpeaks[ dpeaks<np.median(dpeaks)-2.*np.std(dpeaks)]#values much lower than median should not be possible anyway bc of find_peaks
            #res[j]=np.median(dpeaks)
            res[j] = dpeaks
            #plt.figure()
            #plt.plot(ao)
            #plt.plot(peaks, ao[peaks], 'ro')
        res=np.concatenate(res)
        return res/self.samplingRate

        
    def get_frame_periods__PREV(self):
        '''returns the refresh period of the arena screen, not Kinefly
         that result is used to e.g. make the plot pretty
         This is a new method, works better than the old one, but probably breaks when signal gets too noisy (like the old recordings)
         '''
        res = np.zeros(len(self.segments))
        for i,seg in enumerate(self.segments):
            dd = np.diff(seg.data[self.channelmap['arenaOut']])
            zdd = (dd-np.mean(dd))/np.std(dd)
            dist = np.diff(np.where(zdd>3.6)[0])
            res[i] = np.nanmedian(dist) #median distance between significant
        if any(np.isnan(res)):
            print('Warning: found unexpected nans when computing frame periods')
        return res/self.samplingRate

    def stimulusValidationTests(self):
        '''virtual function for base class, fill with meaning in derived classes'''
        pass

    def create_channelmap(self):
        '''this should probably never have to be changed, unless one day the recording channels are switched and mess up the code
         if that happens, make a new, derived base class that overloads this function'''
        if self.nChannels==2:
            self.channelmap_raw = {'wingL':0,'wingR':1,'arenaOut':2,'stimID':3}
        elif self.nChannels==3:
            self.channelmap_raw = {'ephys':0, 'wingL':1,'wingR':2,'arenaOut':3,'stimID':4}
        #make sure var channelmap exists, without overwriting any later changes
        try:
            self.channelmap
        except AttributeError:
            self.channelmap = self.channelmap_raw
        return


    def get_stimulusmap(self, idx):
        '''returns a string that is to be used for labels like plot titles
        takes an int that corresponds to a group_index value, and outputs the corresponding label
         In this base class, always return a not-implemented message
         Remind user to implement it in derived class for each experiment'''
         # this function exists in base class just so code for plotting does not break
         #   should always be implemented in child class
        #self.stimulusmap = {k/10: 'UNKNOWN' for k in range(0,50)}
        #raise Warning('STIMULUS MAP UNKNOWN IN BASE CLASS')
        return 'STIMULUS MAP UNKNOWN IN BASE CLASS'
        
    
    def set_metadata(self, metas, targetIdx=None):
        self.metadata[targetIdx] = metas
        if targetIdx is None: #by default go to the last entry
            targetIdx = len(self.metadata)-1
        #self.flyID[targetIdx] = 'fly'+str(self.metadata[targetIdx]['fly'])+'_'+str(self.metadata[targetIdx]['year'])+str(self.metadata[targetIdx]['month'])+str(self.metadata[targetIdx]['day'])
        cell = self.metadata[targetIdx]['cell'] if 'cell' in self.metadata[targetIdx].keys() else 'NA'
        self.flyID[targetIdx] = 'fly{}_{}_{:04}{:02}{:02}_cell{}'.format(self.metadata[targetIdx]['fly'],self.metadata[targetIdx]['cross'], self.metadata[targetIdx]['year'], self.metadata[targetIdx]['month'], self.metadata[targetIdx]['day'], cell)
        return
                     
    
    # HEREBY FLAGGED FOR REMOVAL
    #def get_by_group(self, idx, dataType='wingDiff'):
    #    return self.get_by_group__all(idx, dataType)
              
    # HEREBY FLAGGED FOR REMOVAL; THERE SHOULD BE NOTHING REFERRING TO IT
    def get_by_group__all(self, idx, dataType='wingDiff'):
        '''Gets all the segments that belong to a certain stimulus type (the group).
          Returns a 2D array of wing diff data from each segment in the group, cut to fit the shortest instance
          (padding by NaN to the longest has been removed)
          Does not do any averaging; for that see get_avg_data()
          This is the functionality formerly provided by __getitem__(self, idx)  (-> that caused obscure problems when using numpy.concatenate)
          '''
        if not isinstance(idx, (int, np.integer)):
            raise TypeError('index needs to be int')
        if idx <= self.group_indices.max():
            temp = self.segments[self.group_indices==idx]
            minLen = min([len(t) for t in temp])
            recipient = np.nan*np.ones(( len(temp),  minLen ))
            for i in range(len(temp)):
                #padding datasegments not necessary when using minLen
                #t = np.pad(temp[i].data[self.channelmap[dataType]], (0,recipient.shape[-1]-temp[i].data[self.channelmap[dataType]].size), 'constant',constant_values=np.nan )
                recipient[i,:] = temp[i].data[self.channelmap[dataType], :minLen]
        return recipient

    def remove_duds(self, GUI=True, *args):
        """ 3 Options:  GUI is True (default) -> go use the manual GUI selection
                          GUI is False but no list given -> use an algorithm to estimate which are wrong
                          GUI is False and a list of items to remove is given -> remove those
          """
        if GUI or len(args)==0:
            self.remove_duds_gui()
        elif not GUI and len(args)==0:
            print("ATTENTION!! You are using an old algorithm to automatically recognize the bad signal, which gives bad results!")
            self.remove_duds_qualityIndex()
        elif not GUI and len(args)==1:
            self.remove_duds_byList(args[0])
        self.average_flywise()
        return
    
    def remove_duds_byList(self, toRemove):
        #self.removedSegs = np.concatenate((self.removedSegs, self.segments[toRemove]))
        #self.removed_groupID = 
        toKeep = np.ones(self.segments.shape, dtype=np.bool)
        toKeep[toRemove] = False
        self.segments = self.segments[toKeep]
        self.group_indices = self.group_indices[toKeep]
        self.keepTrackofCollections = self.keepTrackofCollections[toKeep]
        self.fly_indices = self.fly_indices[toKeep]
        self.flying = self.flying[toKeep]
        #raise NotImplementedError
        self.average_flywise()
    
    def remove_duds_qualityIndex(self):
        ''' Removes the segments where the wing signals are 0 (resp. under a threshold), because they can not be used for analysis. 
           The reasons for this can be a fly not flying or missing detection by Kinefly. Doesn't matter either way.
           '''
        toKeep = np.ones(self.segments.shape, dtype=np.bool)
        for i in range(len(self.segments)):
            quality = np.sum( np.abs(self.segments[i].data[ [self.channelmap['wingL'], self.channelmap['wingR']] ,:])  < self.qualityTheta )
            if quality>self.samplingRate*0.01*2.:#what is left after filtering is too much
                toKeep[i] = False
        self.segments = self.segments[toKeep]
        self.group_indices = self.group_indices[toKeep]
        self.keepTrackofCollections = self.keepTrackofCollections[toKeep]
        self.fly_indices = self.fly_indices[toKeep]
        return
        
    def remove_duds_gui(self):
        ''' opens a simplistic gui that goes trhough all stimulus instances one after another
            buttons to remove or continue '''
        from matplotlib.widgets import Button
        f,ax = plt.subplots()
        done = False
        i=0
        nEl = len(self.segments)
        #select = self.group_indices==args[0]
        #select = self.group_indices>-1
        segs = self.segments
        toRemove = []
        nonflying = []
        dataChannels = [0,1,2] if 'ephys' in self.channelmap.keys() else [0,1]
        #function needed for the selection process as button calbacks
        def _plotNext(event):
            nonlocal i
            nonlocal ax
            nonlocal dataChannels#should be unnecessary
            if i==nEl-1:    return
            i+=1
            if i in toRemove:
                _plotNext(event)
            else:
                ax.clear()
                ax.plot(self.segments[i].data[dataChannels,:].T)
                #ax.set_title("index {!s} stimulus type {!s}: {}".format(i, self.group_indices[i], self.stimulusmap[(self.group_indices[i]+1.)*0.5]) )
                ax.set_title("index {!s} stimulus type {!s}: {}".format(i, self.group_indices[i], self.get_stimulusmap( self.group_indices[i]) ))
                plt.draw()
            return
        def _prev(event):
            nonlocal i
            nonlocal ax
            nonlocal dataChannels#should be unnecessary
            if i<1:     return
            i-=1
            if i in toRemove:
                _prev(event)
            else:
                ax.clear()
                ax.plot(self.segments[i].data[dataChannels,:].T)
                #ax.set_title("index {!s} stimulus type {!s}: {}".format(i, self.group_indices[i], self.stimulusmap[(self.group_indices[i]+1.)*0.5]) )
                ax.set_title("index {!s} stimulus type {!s}: {}".format(i, self.group_indices[i], self.get_stimulusmap(self.group_indices[i]) ))
                plt.draw()
            return
        def _remove(event):
            nonlocal i
            nonlocal toRemove
            if i not in toRemove:
                toRemove.append(i)
            if i<nEl-1:
                _plotNext(event)
            else:
                _prev(event)
            return
        def _close(event):
            nonlocal f
            nonlocal toRemove
            nonlocal nonflying
            plt.close(f)
            del f
            #now actually remove them
            print('to remove:  '+str(toRemove))
            print('not flying:  '+str(nonflying))
            #self.removedSegs = np.concatenate((self.removedSegs, self.segments[toRemove]))
            toKeep = np.ones(self.segments.shape, dtype=np.bool)
            toKeep[toRemove] = False
            self.segments = self.segments[toKeep]
            self.group_indices = self.group_indices[toKeep]
            self.keepTrackofCollections = self.keepTrackofCollections[toKeep]
            self.fly_indices = self.fly_indices[toKeep]
            self.flying[nonflying] = False
            self.flying = self.flying[toKeep]
            #correct the indices of nonFlying for removed trials
            toRemove=np.array(toRemove)
            nonflying = [ nf-sum(toRemove<nf) for nf in nonflying ]
            self.nonFlying = nonflying
            #self.flying = np.delete(self.flying, nonflying)#previous way
            del self._guiButts
            del self._guiFAx
            self.removedIndices = toRemove
            return
            
        def _nonFlying(event):
            nonlocal i
            nonlocal nonflying
            if i not in nonflying:
                nonflying.append(i)
            if i<nEl-1:
                _plotNext(event)
            else:
                _prev(event)
            return
            
        #build the GUI
        axprev = plt.axes([0.5, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.61, 0.05, 0.1, 0.075])
        axremove = plt.axes([0.12, 0.05, 0.1, 0.075])
        axclose = plt.axes([0.75, 0.05, 0.1, 0.075])
        axnonflying = plt.axes([0.23, 0.05, 0.1, 0.07])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(_plotNext)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(_prev)
        bremove = Button(axremove, '\'t is junk')
        bremove.on_clicked(_remove)
        bclose = Button(axclose, 'Close')
        bclose.on_clicked(_close)
        bnonflying = Button(axnonflying, 'not flying')
        bnonflying.on_clicked(_nonFlying)
        #start with the first segment to plot GUI
        ax.plot(segs[0].data[dataChannels,:].T)
        plt.subplots_adjust(bottom=0.2)
        #self._guiButts = resp
        self._guiButts = {'next':bnext, 'prev':bprev, 'rem':bremove, 'close':bclose, 'nonflying':bnonflying }
        self._guiFAx = (f,ax)   
        return        
        
        
    def remove_unreactive(self):
        ''' not used so far, needs elaborating (and rewriting?) '''
        print('NOT FINISHED IMPLEMENTING; NO IDEA WHATS GONNA HAPPEN')
        zscoreTheta = 3.2
        maxArea = self.stimulusPhases_inframes[1] *meanFramePeriod *self.samplingRate
        maxArea = np.arange(int(maxArea-self.samplingRate*0.04), int(maxArea+self.samplingRate*0.04) )
        toKeep = np.ones(self.segments.shape, dtype=np.bool)
        meanFramePeriod = np.mean(self.frame_update_periods)
        for i in range(len(self.segments)):
            # long-pass filter <- leave it
            s = self.segments[i].data['wingDiff']
            # use the z-score as threshold
            value = abs(np.nanmean(s[maxArea]))/np.nanstd(s[:int(18.*meanFramePeriod*self.samplingRate)]) #mean value around expected piece compared to std of beginning
            if value < zscoreTheta:
                toKeep[i] = False
            # determine if maximum is close to the expected location
            
        
        
    def remove_idx(self,idx):
        '''removes the segment given by idx in the segments list. Unlike the '''
        if idx>len(self.segments):
            raise IndexError
        toKeep = np.ones(self.segments.shape, dtype=np.bool)
        toKeep[idx] = False
        self.segments = self.segments[toKeep]
        self.group_indices = self.group_indices[toKeep]
        self.keepTrackofCollections = self.keepTrackofCollections[toKeep]
        self.fly_indices = self.fly_indices[toKeep]
        self.flying = self.flying[toKeep]

    def _adjust_flyIndices_order(self):
        #super(Analyzer, self).sort_by_signal()
        self.fly_indices = self.fly_indices[self.group_indices_sortlist] #leverage possibilities of pandas.Series <- moved away from pd.Series, because of Index problems
        self.flying = self.flying[self.group_indices_sortlist]
        return

        
    def baseline_correct_data(self, **kwargs):
        start = kwargs.get('baseline_startpoint', int(self.preStim*self.samplingRate))
        length = kwargs.get('baseline_length', int(0.4*self.samplingRate))
        for seg in self.segments:
            if 'ephys' in self.channelmap.keys():
                seg.data[self.channelmap['ephys']] -= np.nanmean(seg.data[self.channelmap['ephys'], start:start+length])
            seg.data[self.channelmap['wingDiff']] -= np.nanmean(seg.data[self.channelmap['wingDiff'], start:start+length])
        return 
        
    def make_wingDiffs(self):
        ''' self explanatory '''
        #wingDiffs added to dict of channels
        if 'wingDiff' not in self.channelmap.keys():
            self.channelmap['wingDiff'] = max(self.channelmap.values())+1
        #calculate
        for seg in self.segments:
            wingL = seg.data[self.channelmap['wingL']]
            wingR = seg.data[self.channelmap['wingR']]
            #correct the data to initial baseline
            wingL = wingL-np.mean(wingL[:int(0.05*self.samplingRate)])
            wingR = wingR-np.mean(wingR[:int(0.05*self.samplingRate)])
            #seg.wingDiff = wingL - wingR
            #let data hold the new wingdiff as additional array
            #  test if it exists, else need to stack to array
            if self.channelmap['wingDiff'] >= seg.data.shape[0]:
                seg.data = np.vstack((seg.data, wingL.reshape(1,wingL.size)-wingR.reshape(1,wingR.size) ))
            else:
                seg.data[self.channelmap['wingDiff']] = wingL - wingR
        return
        
    def average_wingDiffs(self):
        return self.average_wingDiffs_flywise()
            
    def average_wingDiffs_all(self):
        '''averaging by group of same stimulus, analog to base class data
        DEPRECATED, FLAGGED FOR REMOVAL
        instead use base class method and access as meanSegments[channelmap['wingDiff']]
        '''
        self.meanWingDiff = {}
        self.stdWingDiff = {}
        for cat in set(self.group_indices):
            #temp=self.get_by_group(cat, 'wingDiff')
            temp = np.squeeze(temp[:, self.channelmap['wingDiff'],:])
            self.meanWingDiff[cat] = np.nanmean(temp, axis=0)
            self.stdWingDiff[cat] = np.nanstd(temp, axis=0)
        return
            
    def average_wingDiffs_flywise(self):
        '''legacy code, could be removed or at least replaced by calls to new functions 
            self.average_flywise()
            self.meanWingDiff=self.get_avg_data('wingDiff')
         '''
        self.meanWingDiff = {}
        self.stdWingDiff = {}
        #average per stim type
        #also average per fly first
        for cat in set(self.group_indices):
            #determine max size required already here to avoid doing the padding twice
            selector = self.group_indices==cat
            tempStim = self.segments[selector]
            flyIDs = self.fly_indices[selector]
            #maxLen = max([len(t) for t in tempStim])
            minLen = min([len(t) for t in tempStim])
            nFlies = len(set(flyIDs))
            minFlyID = flyIDs.min()
            avgRecipient = np.nan*np.ones((nFlies, minLen))
            stdRecipient = np.nan*np.ones((nFlies, minLen))
            flyIDmap = { id: i for i,id in enumerate(set(flyIDs)) } #the order might be mixed up, but doesn't really matter
            for fly in flyIDmap.keys():
                #make a subselection of the stimulus type per fly
                tempFly = tempStim[flyIDs==fly]
                recipient = np.nan*np.ones((len(tempFly), minLen))
                for i in range(len(tempFly)):
                    #pad datasegments if necessary
                    #t = np.pad(tempFly[i].wingDiff, (0,recipient.shape[-1]-tempFly[i].wingDiff.size), 'constant',constant_values=np.nan )
                    recipient[i,:] = tempFly[i].data[self.channelmap['wingDiff'],:minLen]
                #average per fly
                flyIdx = flyIDmap[fly]
                avgRecipient[flyIdx,:] = np.nanmean(recipient, axis=0)
                stdRecipient[flyIdx,:] = np.nanstd(recipient, axis=0)
                #self.meanWingDiff_per_stim_fly[cat][fly] = avgRecipient[fly-minFlyID,:]
            #average over the flies
            self.meanWingDiff[cat] = np.nanmean(avgRecipient, axis=0)
            #gaussian error propagation: Gaussian propagation with y(x1,x2,...)=sum(x_i)/N   s_y = root( sum(dy/dxi_i *s_i)^2 ), where dy/d_xi=1/N for all x_i
            self.stdWingDiff[cat] = np.sqrt( np.nansum(stdRecipient**2.))/nFlies
        return


    def average_by_stimSignal(self):
        super().average_by_stimSignal()
        for k in self.stdSegments.keys():
            self.stdSegments[k] /= np.sqrt( len(set(self.fly_indices[self.group_indices==k])) )

    def average_flywise(self, **kwargs):
        '''average the data first by fly and then all fly means together (avoid double dipping)
        result is stored in meanSegments and stdSegments attributes'''
        self.baseline_correct_data(**kwargs)
        self.meanSegments = {}
        self.stdSegments = {}    
        for cat in set(self.group_indices):
            selector = self.group_indices==cat
            tempStim = self.segments[selector]
            flyIDs = self.fly_indices[selector]
            #determine max size required already here to avoid doing the padding twice
            #maxLen = max([len(t) for t in tempStim])
            minLen = min([len(t) for t in tempStim])
            nFlies = len(set(flyIDs))
            avgRecipient = np.nan*np.ones((nFlies, len(self.channelmap), minLen))
            stdRecipient = np.nan*np.ones((nFlies, len(self.channelmap), minLen))
            flyIDmap = { id: i for i,id in enumerate(set(flyIDs)) } #the order might be mixed up, but doesn't really matter
            for fly in flyIDmap.keys():
                #make a subselection of the stimulus type per fly
                tempFly = tempStim[flyIDs==fly]
                recipient = np.nan*np.ones((len(tempFly), len(self.channelmap), minLen))
                for i in range(len(tempFly)):
                    #pad datasegments if necessary because the recording does not stop at exactly the same moment
                    #t = np.pad(tempFly[i].data, ((0,0),(0,maxLen-tempFly[i].data.shape[1])), 'constant',constant_values=np.nan )
                    recipient[i,:,:] = tempFly[i].data[:,:minLen]
                flyIdx = flyIDmap[fly]
                avgRecipient[flyIdx,:,:] = np.nanmean(recipient, axis=0)
                stdRecipient[flyIdx,:,:] = np.nanstd(recipient, axis=0)
            #average over the flies per stim category
            self.meanSegments[cat] = np.nanmean(avgRecipient, axis=0)
            #gaussian error propagation: Gaussian propagation with y(x1,x2,...)=sum(x_i)/N   s_y = root( sum(dy/dxi_i *s_i)^2 ), where dy/d_xi=1/N for all x_i
            self.stdSegments[cat] = np.sqrt( np.nansum(stdRecipient**2., axis=0)/(nFlies**2))
        return
        
    
    def get_avg_data(self, dataType):
        '''name self-explanatory, 
            returns a dicts, with each entry a stimulus number, keys are integer indices ''' 
        #self.average_wingDiffs_flywise()
        #return (self.meanWingDiff, self.stdWingDiff)
        try:
            self.meanSegments
        except AttributeError:
            self.average_flywise()
        return { k: v[self.channelmap[dataType]] for k,v in  self.meanSegments.items() }
        
        
    def get_std_data(self, dataType):
        return { k: v[self.channelmap[dataType]] for k,v in  self.stdSegments.items() }
    
    def plotMean(self, dataType='wingDiff'):
        """default argument remains for legacy reasons"""
        #self.average_wingDiffs()
        #self.average_flywise()
        try:
            self.meanSegments
        except AttributeError:
            self.average_flywise()
        #f,ax = plt.subplots(2,3)
        #ax = ax.flatten()
        meanWingDiff = self.get_avg_data(dataType)
        stdWingDiff = self.get_std_data(dataType)
        specs = strategies.RectangularStrategy().get_grid(self.group_indices.max()-self.group_indices.min()+1)
        f=plt.gcf()
        ax=[f.add_subplot(sp) for sp in specs]
        for k in meanWingDiff.keys():
            ax[k].plot(self.timeax[:meanWingDiff[k].size], meanWingDiff[k])
            ax[k].fill_between(self.timeax[:meanWingDiff[k].size], meanWingDiff[k]+stdWingDiff[k] , meanWingDiff[k]-stdWingDiff[k], alpha=0.5)
            sel = self.group_indices==k
            #ax[k].set_title(self.stimulusmap[ np.round(self.segments[sel][0].meanValue, decimals=1)])
            ax[k].set_title(self.get_stimulusmap(k))
        self.meanFigure = f
        self.prettify_plot()
        return


    def confirm_flyIDs(self):
        '''
        is supposed to come into play when two collection are combined
        compares the metadata to come to a conclusion whether two collections are from the same animal or not
        attributes fly_indices accordingly
        '''
        #check if a certain ID appears more than once
        #be careful in case there is more than one fly that appears double
        vals = list(self.flyID.values())
        keys = list(self.flyID.keys())
        if not len(vals)==len(set(vals)):
            #set higher ID values to lower one
            for i,setEl in enumerate(set(vals)):
                if vals.count(setEl)==1:
                    continue
                wrongIdx = [j for j,el in enumerate(vals) if el==setEl ]
                #recurse or iterate to reduce the wrongIdx ones
                ## go from back through and displace the higher to the lower, first index, then remove entry from dict
                k=len(wrongIdx)-1
                while k>0:
                    assert(vals[wrongIdx[k]]==vals[wrongIdx[k-1]])#just make sure during debugging I'm not mistaken
                    toReplace = keys[wrongIdx[k]]
                    replaceBy = keys[wrongIdx[k-1]]
                    #for same animal, replace the higher number ones by the lower one 
                    self.fly_indices[self.fly_indices==toReplace] = replaceBy
                    #remove entry from flyID dict
                    del self.flyID[toReplace]
                    #shift the remaining one down so that there are no unused slots in the sequence
                    self.fly_indices[self.fly_indices > toReplace] = self.fly_indices[self.fly_indices>toReplace] -1
                    self.flyID = { (k if k<toReplace else k-1 ):v  for k,v in self.flyID.items() } #yes, complex dict-comprehension, I know... Shifts the key one down if key is bigger than what got removed
                    k-=1
        #check if all fly_indices have a number that corresponds to a flyID
        #check vice-versa, if all flyIDs occur in some fly_indices
        intIdx_set = set(self.fly_indices)
        if intIdx_set != set(self.flyID.keys()):
            #needReorder = True
            keySet = set(self.flyID.keys())
            #differentiate between the two issues
            #is one of them longer, then there is prob the issues
            #is one of them missing a piece of the sequence, then correct that
            #in other terms there are three cases: A subset of B, B subset of A, they have different elements (can even be same length)
            if keySet < intIdx_set: #keySet is a strict subset of intIdx_set (not equal)
                #don't know how to correct this without more information
                warnings.warn('Warning: There are more different fly_indices than in flyIDs dict')
            elif intIdx_set < keySet: #inverse case
                #don't know how to correct this without more information
                warnings.warn('Warning: There are more different flyIDs than fly_indices')
                if not len(vals)==len(set(vals)):
                    warnings.warn('Warning: There appear to be duplicate flies in the flyID dict')
            else:
                if len(intIdx_set)==len(keySet):
                    ans=input('There seems to be a simple mismatch between the flyID dict and the fly_indices array.\nThe situation s the following: ' +
                    '\nflyID {0} \n set of fly_indices {1}' +
                    '\nShould we try to correct this by simply setting one to be the other? (y/n)'.format( str(list(self.flyID.keys())), str(set(self.fly_indices)) ) )
                    if ans.lower()=='n':
                        return
                    elif ans.lower()=='y':
                        #determine which is more probable to be wrong and replace that by the other
                        #hint what could be wrong is the sequential order (is there a jump?)
                        keySetTup = tuple(keySet) #needed because sets don't support indexing
                        anyJumpIdx = [0] if keySetTup[0]>0 else []
                        anyJumpIdx.extend([i for i in range(1,len(keySetTup)) if  keySetTup[i]-keySetTup[i-1]>1])
                        if anyJumpIdx:
                            anyJumpSize = [keySetTup[0] if anyJumpIdx[0]==0 else [] ]
                            anyJumpSize.extend( [keySetTup[idx]-keySetTup[idx-1] for idx in anyJumpIdx ])
                            for ix,sz in reverse(zip(anyJumpIdx,anyJumpSize)):
                                self.flyID = { k if k<ix else ix-sz+1 : val for k,val in self.flyID.items() }
                    else:
                        warnings.warn('Warning: unknown answer possibility, ignored')
        return #that's all i can come up with right now how to correct for mistakes and stuff
    
    def plot_stimPhases(self, *args):
        pass

    def prettify_plot(self, *args):
        if len(args)==0:
            try:
                self.meanFigure
            except AttributeError:
                self.plotMean()
            #get all axes
            ax = self.meanFigure.axes
        else:
            ax = args[0].axes
        #adjust all ylims to maximum required
        maxNec = max([ a.lines[0].get_ydata().max() for a in ax if any(a.lines)])
        minNec = min([ a.lines[0].get_ydata().min() for a in ax if any(a.lines)])
        for a in ax:
            a.set_ylim([minNec*1.1, maxNec*1.2])
            a.set_xlim(a.lines[0].get_xdata()[[0,-1]])
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
        ax[0].set_ylabel('wingbeat L-R [radian]')
        ax[4].set_ylabel('wingbeat L-R [radian]')
        [ax[i].set_xlabel('time [s]') for i in range(4,8)]
        plt.suptitle(self.fileOrigin[0][:-10])
        self.plot_stimPhases()
        return        
        
    def plot_by_stimSignal(self, dataType):
        '''Makes a figure with one subplot per stimulus condition (type), and each subplot shows all the individual data segments of that stimulus type.
         Added in black is the average of all segments'''
        assert(dataType in self.channelmap.keys())
        #self.average_wingDiffs()
        #self.average_flywise()
        try:
            self.meanSegments
        except AttributeError:
            self.average_flywise()
        nCols = len(set(self.group_indices)) #is different from self.group_indices.max() when all instances of one group get removed or merged to other fly
        #fig, ax = plt.subplots( int(self.nChannels//2+0.5), nCols)
        specs = strategies.RectangularStrategy().get_grid(self.group_indices.max()-self.group_indices.min()+1)
        f=plt.gcf()
        ax=[f.add_subplot(sp) for sp in specs]
        #if int(self.nChannels/2.+0.5)<2:
        #    ax=[ax]
        #plt.figure()
        for i in set(self.group_indices):
            #plotIdx = (i//4)*4+i+1
            temp=self.get_by_group(i)[:,self.channelmap[dataType]]
            #temp = np.squeeze(temp[:, self.channelmap['wingDiff'],:])
            for t in temp:
                #plt.subplot(2,4,i+1)
                # plot wingbeat L and R
                ax[i].plot(self.timeax[:t.size], t, alpha=0.6)
                #plt.plot(t)
            #plt.subplot(2,4,i+1)
            #plt.plot(self.meanWingDiff[i], color='k')
            ax[i].plot(self.timeax[:self.meanSegments[i].shape[1]], self.meanSegments[i][self.channelmap[dataType]], color='k')
            ax[i].set_title(self.get_stimulusmap(i))
        #plt.subplot(2,4,1)
        ax[0].set_ylabel(dataType)
        #plt.ylabel('wingbeat diff L-R')
        plt.suptitle(self.fileOrigin[0][:-10])
        return f
        
    def plot_certain_stimgroup(self, idx, dataType=None):
        '''make a figure with one plot for only the type of stimulus given by the index'''
        #assert(dataType in self.channelmap.keys())
        #self.average_wingDiffs()
        #self.average_flywise()
        try:
            self.meanSegments
        except AttributeError:
            self.average_flywise()
        meanWingDiff = self.meanSegments[idx][self.channelmap['wingDiff']]
        stdWingDiff = self.stdSegments[idx][self.channelmap['wingDiff']]
        #temp = self.get_by_group(idx)
        #k = idx
        #plt.plot(self.timeax[:self.meanWingDiff[k].size], self.meanWingDiff[k])
        if self.nChannels==2:
            f, ax = plt.subplots(1,1)
            ax=[ax]
            ax[0].plot(self.timeax[:meanWingDiff.size], meanWingDiff)
            #plt.fill_between(self.timeax[:self.meanWingDiff[k].size], self.meanWingDiff[k]+self.stdWingDiff[k] , self.meanWingDiff[k]-self.stdWingDiff[k], alpha=0.5)
            #plt.title(self.stimulusmap[(k+1.)*0.5])
            ax[0].title(self.get_stimulusmap(k))
            ax[0].ylabel('wingbeat L-R [radian]')
            ax[0].xlabel('time [s]')
            ax[0].xlim([0, self.timeax[meanWingDiff.size] ])
        elif self.nChannels==3:
            f, ax = plt.subplots(2,1)
            ax[0].plot(self.timeax[:meanWingDiff.size], meanWingDiff)
            ax[0].fill_between(self.timeax[:meanWingDiff.size], meanWingDiff+stdWingDiff , meanWingDiff-stdWingDiff, alpha=0.5)
            ax[0].set_ylabel('wingbeat L-R [radian]')
            ax[0].set_xlim([0,self.timeax[meanWingDiff.size]])
            ax[1].plot(self.timeax[:self.meanSegments[idx][self.channelmap['ephys']].size], self.meanSegments[idx][self.channelmap['ephys']])
            ax[1].set_ylabel('voltage [V]')
            ax[1].set_xlabel('time[s]')
            ax[1].set_xlim([0,self.timeax[meanWingDiff.size]])
        self.plot_stimPhases(f)
        ax[0].set_title(self.get_stimulusmap(idx))
        return plt.gca()

    
    def plot_stimGroups2confirm(self, ax_rawdataAll):
        gridx = self.group_indices
        numel=len(gridx)
        cs = ['b','r','g','y','k', 'c', ]
        for i in gridx:
            temp = scl.segments[gridx==i]
            for t in temp:
                timeBounds = t.boundaries
                ax_rawdataAll.plot(scl.timeax[timeBounds[0]:timeBounds[1]]/self.samplingRate, t.data[self.channelmap['stimID'],:], color=cs[i])
        return
        

    def __add__(self, otherCollection):
        '''overloaded to get a string-like functionality, i.e. using '+' will append the second to the first and rerun the necessary steps to order'''
        nFiles = max(self.fileOrigin.keys())
        #nOtherCollections = len(otherCollection.fileOrigin)
        #self.fileOrigin[nFiles+1:nFiles+1+nOtherCollections] = otherCollection.fileOrigin[:]
        self.fileOrigin.update({ nFiles+1+otherKey: otherVal  for otherKey,otherVal in otherCollection.fileOrigin.items() })
        #otherCollection.remove_duds()
        super(BaseAnalyzer, self).__add__(otherCollection)
        #need to implement that the flyIDs get updated for all collections (consider that other collection also can have more than one
        #shift the indices of dict entries
        toAdd = max(self.flyID.keys())+1
        tempOtherFlyID = { k+toAdd: val for k,val in otherCollection.flyID.items() }
        self.flyID.update(tempOtherFlyID)
        #yes, only do this here, because I want to track that I addded these flyIDs
        if len(otherCollection)==0:
            return self #nothin to do, skip rest because assertion downstream causes problems
        tempOtherFlyIndices = otherCollection.fly_indices+toAdd
        self.fly_indices = np.concatenate((self.fly_indices, tempOtherFlyIndices))
        self.flying = np.concatenate((self.flying, otherCollection.flying))
        self._adjust_flyIndices_order()
        self.reallocate_fly_indices()
        assert(len(self.flyID.keys())-1==max(self.fly_indices))
        # indices need to be shifted to accmomodate all without overlap, and then reconfirm that identical animals get only one index
        self.confirm_flyIDs()
        self.average_flywise()
        return self
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def add_collection(self, other):
        '''Append the second to the first with all its segments and rerun the necessary steps to order
            exactly equivalent to instance.__add__(other) '''
        #add the two collections using baseclass method
        #then recalculate the wingbeats and stuff
        #keep track of which segments are from which collection (?)
        self.__add__(other)
        return

    def print_N_stims_per_fly(self):
        groupIdx = set(self.group_indices)
        flyIdx = set(self.fly_indices)
        try: 
            self.stimulusmap #when there is a child class providing infos
            separator = "\t\t" #need more space to be pretty
            firstLine = separator+"\t".join([self.get_stimulusmap(gix) for gix in groupIdx])
        except AttributeError:
            separator = "\t"
            firstLine = separator+"\t".join(groupIdx)
        print(firstLine)
        for fix in flyIdx:
            print(self.flyID[fix]+separator+separator.join([str(sum(self.group_indices[self.fly_indices==fix]==gx)) for gx in groupIdx ]) )
    
    def save_collection(self, filename, path='.'):
        import pickle
        with open(os.path.normpath(path+'/'+filename), 'rb') as f:
            pickle.dump(self, f)
        return
    


    def convert2neo(self):
        import neo
        import quantities as pq
        #allfields = vars(self).copy()
        allfields = [v for v in dir(self) if not callable(getattr(self,v))]
        allfields = {v:getattr(self,v) for v in allfields }

        #remove attributes that are taken care of separately
        deleteFields = ['segments', 'meanSegments', 'removedSegs', 'stdSegments', 'meanFigure', '__doc__', '__weakref__', '__dict__']
        for thisfield in deleteFields:
            try:
                del allfields[thisfield]
            except KeyError:
                continue #or pass, simply carry on
        allfields['fly_indices'] = allfields['fly_indices'].astype(int)
        allfields['group_indices'] = allfields['group_indices'].astype(int)
        allfields['keepTrackofCollections'] = allfields['keepTrackofCollections'].astype(int)
        if allfields['group_indices_sortlist'].__class__.__module__ != 'numpy':
            allfields['group_indices_sortlist'] = allfields['group_indices_sortlist'].to_numpy(dtype=int)
        else:
            allfields['group_indices_sortlist'] = allfields['group_indices_sortlist'].astype(int)
        
        neoblock = neo.Block(**allfields)
        for i,seg in enumerate(self.segments):
            neoseg = neo.Segment()
            ch=seg.data
            if ch.shape[0] < ch.shape[1]:
                ch=ch.T
            segdict = vars(seg).copy()
            del segdict['data']
            segdict['samplingRate'] = self.samplingRate
            #store corresponding entry in fly_indices and group_indices as string in name attribute
            neoanalog = neo.AnalogSignal(ch*pq.V, sampling_rate=self.samplingRate*pq.Hz, name="flyIndex {},groupIndex {},flying {}".format(self.fly_indices[i], self.group_indices[i], int(self.flying[i])), **segdict)
            neoseg.analogsignals.append(neoanalog)
            neoblock.segments.append(neoseg)
        #neoseg=neo.Segment()
        #neoseg.append(neo.AnalogSignal(self.timeax*pq.S, ))
        #neoblock.annotations['classtype'] = type(self)
        neoblock.annotations['classtype'] = self.__class__.__name__
        return neoblock
        
    def write2nix(self, filename):
        neoblock = self.convert2neo()
        #self.replace_numeric_keys(neoblock.annotations) #inplace, because dict is mutable
        # add info about original fieldtypes for recovery later
        #fieldtypes = {name: type(field) for name, field in neoblock.annotations.items()}
        #neoblock.annotations['_FIELDTYPES'] = fieldtypes
        for k,val in neoblock.annotations.items():
            if isinstance(val, np.ndarray):
                neoblock.annotations[k] = self.recurse_encode_list(val.tolist())
        #for k,val in neoblock.annotations.items():
        #    if isinstance(val, list):
        #        neoblock.annotations[k] = self.recurse_encode_list(val)
        #for k,val in neoblock.annotations.items():
        #    if isinstance(val, dict):
        #        neoblock.annotations[k] = self.recurse_encode_dict(val)
        print('Hello')
        #pdb.set_trace()
        #deleteFields = ['__module__', 'afterStim', 'channelLabels', 'channelmap', 'channelmap_raw', 'fileOrigin', 'flyID', 'fly_indices', 'flying', 'frame_update_periods', 'group_indices', 'group_indices_sortlist', 'keepTrackofCollections', 'meanFramePeriod', 'metadata', 'nChannels', 'originalOrder', 'preStim', 'qualityTheta', 'samplingRate', 'stimulusPhases_inframes', 'stimulus_period_inframes', 'stimulusmap', 'timeax']
        #deleteFields = ['__module__', 'channelLabels', 'channelmap', 'channelmap_raw', 'fileOrigin', 'flyID', 'fly_indices', 'flying', 'frame_update_periods', 'group_indices', 'group_indices_sortlist', 'keepTrackofCollections', 'meanFramePeriod', 'metadata', 'nChannels', 'originalOrder', 'preStim', 'qualityTheta', 'samplingRate', 'stimulusPhases_inframes', 'stimulus_period_inframes', 'stimulusmap', 'timeax']
        #for field in deleteFields:    del neoblock.annotations[field]
        from neo.io import NixIO
        try:
            neofile = NixIO(filename=filename)
            neofile.write_block(neoblock)
        finally:
            neofile.close()
        return

    def write2neomatfile(self, filename):
        neoblock = self.convert2neo()
        #workaround for dicts with numbers for keys
        self.replace_numeric_keys(neoblock.annotations) #inplace, because dict is mutable
        from neo.io import NeoMatlabIO
        neofile = NeoMatlabIO(filename=filename)
        neofile.write_block(neoblock)
        self.retrieve_numeric_keys(vars(self))
        return


    def convertFromNeo(self, neoblock):
        #extract the segments as instances of class Segment
        #instance all the  class attributes that are defined in the construcotr from block's annotation
        for key, value in neoblock.annotations.items():
            setattr(self, key, value) 
        Fs = neoblock.annotations['samplingRate']
        #self.segments = np.array([ Segment(seg['boundaries'], seg['meanValue'], seg['data'], Fs=Fs)
        #                          for seg in neoblock.segments ])
        segments = [None]*len(neoblock.segments) #preallocate
        for i,seg in enumerate(neoblock.segments):
            signal = seg.analogsignals[0]
            if signal.shape[0]>signal.shape[1]: #expect this to be the case, because Matlab style
                signal = signal.T
            annots = signal.annotations
            segments[i] = Segment(annots['boundaries'], annots['meanValue'], np.array(signal) , Fs=Fs)
        self.segments = np.array(segments) #convert and store
        return
    
    
    @classmethod
    def retrieveFromNeomatfile(classtype, neoFilename):
        from neo.io import NeoMatlabIO
        neofile = NeoMatlabIO(filename=neoFilename)
        neoblock = neofile.read_block()
        #classtype = neoblock.annotations['classtype']
        self = classtype.__new__(classtype)
        self.convertFromNeo(neoblock)
        #now convert all the shitty scipy.io.matlab.mio5_params.mat_struct to a dict
        self.convert_scipyStructs(vars(self)) #yes, it should do this in-place, __dict__ is linked to vars(self)
        # also can do classtype.convert_scipyStructs(...)
        self.retrieve_numeric_keys(vars(self))
        #apparently the line above doesn't work completely "in the wild", through probly the error was in encoding
        self.retrieve_numeric_keys(self.fileOrigin)
        self.retrieve_numeric_keys(self.stimulusmap)
        self.retrieve_numeric_keys(self.flyID)
        self.retrieve_numeric_keys(self.metadata)
        # assert that critical variables are in array datatype
        for key in vars(self).keys():
            if key in ('group_indices', 'fly_indices','keepTrackofCollections','flying'):
                setattr(self, key, np.array([getattr(self, key)]).flatten() )
        return self
    
    @classmethod
    def retrieveFromNix(classtype, neoFilename):
        from neo.io import NixIO    
        neofile = NixIO(filename=neoFilename)
        neoblock = neofile.read_block()
        #pdb.set_trace()
        neofile.close()
        #classtype = neoblock.annotations['classtype']
        self = classtype.__new__(classtype)
        self.convertFromNeo(neoblock)
        #self.retrieve_numeric_keys(vars(self))
        return self
    
    @staticmethod
    def num_to_text(numeral):
        '''helper function to transform dict keys that are numericals (int, float) to valid matlab struct field names'''
        return 'strAsNum_'+ str(numeral).replace('.','_')
    
    @staticmethod
    def text_to_num(numstring):
        '''helper function to transform certain struct fieldnames into numerical dict keys'''
        if not numstring[:9]=='strAsNum_':
            print('not correct format')
            return numstring
        numstring = numstring[9:]
        if numstring.count('_')==0:
            return int(numstring)
        else:
            return float(numstring.replace('_','.'))
        return
    
    @staticmethod
    def replace_numeric_keys(d):
        '''helper to take a dict and replace the keys that are numerics by string versions'''
        keys = list(d.keys())
        for k in keys: #annotations is a dict
            v = d[k]
            if isinstance(v, dict): #nested dict
                BaseAnalyzer.replace_numeric_keys(v)
            if isinstance(k, (int, float, np.number)):
                d[BaseAnalyzer.num_to_text(k)] = v
                del d[k]
        return
        
    @staticmethod
    def retrieve_numeric_keys(dc):
        '''helper to invert the operation done by replace_numeric_keys'''
        for k,v in tuple(dc.items()):
            if isinstance(v, dict):
                BaseAnalyzer.retrieve_numeric_keys(v)
            if isinstance(k, str) and 'strAsNum_' in k:
                dc[BaseAnalyzer.text_to_num(k)] = v
                del dc[k]
        return
        
    @staticmethod
    def convert_scipyStructs(dc):
        '''helper to convert scipy io mat_struct objects to python dicts recursively, input is a dict (e.g. vars(self)'''
        import scipy
        for varname,varvalue in dc.items():
            if isinstance(varvalue, scipy.io.matlab.mio5_params.mat_struct):
                newvalue = {k: getattr(varvalue, k) for k in varvalue._fieldnames}
                dc[varname] = newvalue
                BaseAnalyzer.convert_scipyStructs(dc[varname])
            elif isinstance(varvalue, dict):
                BaseAnalyzer.convert_scipyStructs(varvalue) #no assignment varvalue= convert... necessary, dict is mutable
        return
    

    @staticmethod
    def recurse_encode_dict(d):
        l=['__dictAlias_N'+str(len(d))]
        for k,v in d.items():
            l.append(k)
            if isinstance(v, dict):
                l.extend(BaseAnalyzer.recurse_encode_dict(v))
            elif isinstance(v, list):
                l.extend(BaseAnalyzer.recurse_encode_list(v))
            else:
                l.append(v)
        return l
        
    @staticmethod
    def recurse_decode_dictAlias(aliasList, startAt=0):
        d=dict()
        if not aliasList[startAt].startswith('__dictAlias_N'):
            return None
        windForward=0 #number of elements processed (including in subroutines)
        nItems = int(aliasList[startAt].split('_N')[1])
        windForward += nItems*2
        
        nItems = nItems*2 +1+startAt #each dict entry corresponds to 2 list entries and offset for first element
        i = startAt+1
        while i<nItems:
            #pdb.set_trace()
            newkey = aliasList[i]
            newval = aliasList[i+1]
            if isinstance(newval, str):
                if newval.startswith('__dictAlias_N'):
                    #newval = recurse_decode_dictAlias(aliasList[i+1:])
                    newval, jumpForward = BaseAnalyzer.recurse_decode_dictAlias(aliasList, startAt=i+1)
                    windForward += jumpForward
                    i += jumpForward
                    nItems += jumpForward
                    #nItems += len(newval)*2 
                    #i += len(newval)*2
                elif newval.startswith('__listAlias_N'):
                    #newval = recurse_decode_listAlias(aliasList[i+1:])
                    newval, jumpForward = BaseAnalyzer.recurse_decode_listAlias(aliasList, startAt=i+1)
                    windForward += jumpForward
                    i += jumpForward
                    nItems += jumpForward
            d[newkey] = newval
            i+=2
        #optional check for completeness
        if startAt==0: #simple basic call
            return d
        else: #nested calls
            #return new list and the amount of elements processed
            return d, windForward

    @staticmethod
    def recurse_encode_list(l):
        newlist = ['__listAlias_N'+str(len(l))]
        #newlist.extend([None]*len(l))
        i=0
        #nextItem=0 # next index to write
        while i<len(l):
            if isinstance(l[i], dict):
                newval = BaseAnalyzer.recurse_encode_dict(l[i])
                newlist.extend(newval)
                #nextItem += len(newval)
            elif isinstance(l[i], (list,np.ndarray)):
                newval = BaseAnalyzer.recurse_encode_list(l[i])
                newlist.extend(newval)
                #nextItem += len(newval)
            else:
                newlist.append(l[i])
            i+=1
        return newlist
        
    
    @staticmethod
    def recurse_decode_listAlias(aliasList, startAt=0):
        if not aliasList[startAt].startswith('__listAlias_N'):
            return None
        #if startAt!=0: #if called recursively (in subroutine)
        windForward=0 #number of elements processed (including in subroutines)
        nItems = int(aliasList[startAt].split('_N')[1])
        #windForward += nItems+1
        windForward += nItems
        #shortcut
        if not (any(['Alias' in el for el in aliasList[startAt+1:startAt+1+nItems] if isinstance(el,str)]) ): #no sub-lists, -dicts or other
            newlist = aliasList[startAt+1:startAt+1+nItems]
        #full process
        else:
            newlist = [None]*nItems
            nItems = nItems+1+startAt#add offset for loop test-variable
            i = startAt+1
            nextItem=0 #index of newlist element to set next
            while i<nItems:
                newval = aliasList[i]
                #pdb.set_trace()
                if isinstance(newval, str):
                    if newval.startswith('__listAlias_N'):
                        #newlist[nextItem] = recurse_decode_listAlias(aliasList[i:])
                        newval, jumpForward = BaseAnalyzer.recurse_decode_listAlias(aliasList, startAt=i)
                        #jumpForward -= 1
                        windForward += jumpForward
                        i += jumpForward
                        nItems += jumpForward
                    elif newval.startswith('__dictAlias_N'):
                        newval, jumpForward  = BaseAnalyzer.recurse_decode_dictAlias(aliasList[i:])
                        windForward += jumpForward
                        i += jumpForward
                        nItems += jumpForward
                newlist[nextItem] = newval
                i += 1
                nextItem += 1
            #newlist = aliasList[1:1+nItems]
        if startAt==0: #simple basic call
            return newlist
        else: #nested calls
            #return new list and the amount of elements processed
            return newlist, windForward
    
    def clean_flyIDs(self):
        idkeys = list(self.flyID.keys())
        indicesSet = set(self.fly_indices)
        for k in idkeys:
            if not k in indicesSet:
                del self.flyID[k]
        return
    
    def reallocate_fly_indices(self):
        keyList = sorted(set(self.fly_indices)) #must be small to big
        prev=-1
        for k in keyList:
            if k-prev>1:
                new_k = prev+1
                self.flyID[new_k] = self.flyID[k]
                del self.flyID[k]
                self.fly_indices[self.fly_indices==k] = new_k
                k = new_k
            prev = k
        return
    
    def deduplicate_IDs(self):
        keylist = sorted(list(self.flyID.keys()))
        vallist = [self.flyID[k].lower() for k in keylist]
        print(keylist,vallist)
        for k,v in zip(keylist,vallist):
            print(k,v)
            for kp,vp in zip(keylist[k+1:],vallist[k+1:]):
                if v==vp:
                    del self.flyID[kp]
                    keylist.remove(kp)
                    print(keylist)
        return
