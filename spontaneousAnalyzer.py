import baseClasses as bc
import importlib
#importlib.reload(bc)


class DummyAnalyzer(bc.BaseAnalyzer):
    LoomingWB_version = '1.0.4'#to keep track of class versions, makes easier to check for changes. Starts with 1.0.0 on 18.1.2020
    """version 1.0.4 changes timeax such that time coordinate 0  is at the end of looming"""
    stimulusmap = {0.5:'loom left rot. clw', 1.0:'loom left rot. cck', 1.5:'loom left no rot.',
                  2.0:'loom right rot. clw', 2.5:'loom right rot. cck', 3.0:'loom right no rot.',
                  3.5:'no loom rot. clw', 4.0:'no loom rot. cck'}
    stimulusPhases_inframes = [0,47,60,74] #frame numbers where stuff changes
    stimulus_period_inframes = 60
    preStim = 0.1 #s
    afterStim = 0.25 #s
    kineflyPeriod=0.02
    
    #def __init__(self, fpath, *args, **kwargs):
    def __init__(self, loader, *args, **kwargs):
        #loader = bc.DataLoader(fpath)
        fpath = loader.file
        data, samplingRate = loader.get_data()
        samplingRate = round(samplingRate/100.,0)*100.

        stimBound = loader.get_stimulusBoundaries()
        stimBound[0::2] = stimBound[0::2]-self.preStim*samplingRate
        stimBound[1::2] = stimBound[1::2]+self.afterStim*samplingRate
        #newBounds = np.array(zip(stimBound[1::2], stimBound[0::2]))
        #newBounds = np.concatenate(([0],stimBound,[data.shape[1],]))
        newBounds = np.concatenate(([0], stimBound[:], [data.shape[1]]))
        #lastPiece = newBounds[-1]-newBounds[-2]
        lastPiece=1000
        print(lastPiece)
        data[4] = np.concatenate(( np.zeros(newBounds[1]-2*lastPiece),np.ones(lastPiece), np.zeros(lastPiece), 
                                  np.concatenate([np.concatenate(( np.zeros(newBounds[i]-newBounds[i-1]) ,np.zeros(newBounds[i+1]-newBounds[i]-2*lastPiece),np.ones(lastPiece),np.zeros(lastPiece))) for i in range(2,newBounds.size-1,2)])
                                  #, np.ones(lastPiece), np.zeros(newBounds[-1]-newBounds[-2]) #somehow too much
                                  ))
        metas = loader.get_metadata()
        super(DummyAnalyzer, self).__init__(newBounds, data, Fs=samplingRate, fname=fpath.split('/')[-1], metadata=metas, **kwargs)
        self.timeax = self.timeax-self.preStim-self.stimulusPhases_inframes[1]*self.meanFramePeriod #center 0 on end of loom
        
