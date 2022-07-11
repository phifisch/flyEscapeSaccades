import baseClasses as bc
#import importlib
#importlib.reload(bc)
#from pwctools.pwc_medfiltit import pwc_medfiltit as medianfilter
from scipy.signal import medfilt as medianfilter

plt = bc.plt


class DirectionTuning_Analyzer(bc.BaseAnalyzer):
    DirTuning_version = '1.0.0'#to keep track of class versions, makes easier to check for changes. Starts with 1.0.0 on 18.1.2020
    stimulusmap = {1.0: 'CLW', 2.0:'CCK', 3.0:'DOWN', 4.0:'UP'}
    stimulusPhases_inframes = [0,64*4+30]
    preStim = 0.2 #s
    afterStim = 0.5 #s
    
    def __init__(self,fpath, *args,**kwargs):
        loader = bc.DataLoader(fpath)
        data, samplingRate = loader.get_data()
        filterwidth = 51 if samplingRate>1500 else 9
        if data.shape[0]==5:
            data[1,:] = medianfilter(data[1,:], filterwidth)
            data[2,:] = medianfilter(data[2,:], filterwidth)
        elif data.shape[0]==4:
            data[0,:] = medianfilter(data[0,:], filterwidth)
            data[1,:] = medianfilter(data[1,:], filterwidth)
        stimBound = loader.get_stimulusBoundaries()
        
        stimBound[0::2] = stimBound[0::2]-self.preStim*samplingRate
        stimBound[1::2] = stimBound[1::2]+self.afterStim*samplingRate
        metas = loader.get_metadata()
        super(DirectionTuning_Analyzer, self).__init__(stimBound, data, Fs=samplingRate, fname=fpath.split('/')[-1], metadata=metas, **kwargs)
        self.timeax = self.timeax - self.preStim
        self.stimulusPhases_inframes.insert(0,self.stimulusPhases_inframes[0]-self.preStim/self.meanFramePeriod)
        self.stimulusPhases_inframes.append(self.stimulusPhases_inframes[-1]+self.afterStim/self.meanFramePeriod)
    
    def get_stimulusmap(self, idx):
        '''returns a string that is to be used for labels like plot titles
         takes an int that corresponds to a group_index value, and outputs the corresponding label
         The transform from int index to voltage value used as key in the stimulusmap dict is subclass specific'''
        return self.stimulusmap[idx+1]
    
    def stimulusValidationTests(self):
        pass
    
    def baseline_correct_data(self,**kwargs):
        length = kwargs.get('length', 0.2*self.samplingRate if self.preStim>=0.2 else self.preStim*self.samplingRate)
        length = int(length)
        start = kwargs.get('startpoint',self.preStim*self.samplingRate-length)
        start = int(start)
        super(DirectionTuning_Analyzer, self).baseline_correct_data(startpoint=start, length=length)
        return
        
    def compareOpposites(self, dataType):
        avg = self.get_avg_data(dataType)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.timeax[:avg[0].size], avg[0])
        plt.plot(self.timeax[:avg[1].size], avg[1])
        plt.legend([self.get_stimulusmap(0), self.get_stimulusmap(1)])
        plt.subplot(2,1,2)
        plt.plot(self.timeax[:avg[2].size], avg[2])
        plt.plot(self.timeax[:avg[3].size], avg[3])
        plt.legend([self.get_stimulusmap(2),self.get_stimulusmap(3)])
        