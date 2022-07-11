#from baseClasses import BaseAnalyzer, DataLoader
import baseClasses as bc
#import importlib
#importlib.reload(bc)
#from pwctools.pwc_medfiltit import pwc_medfiltit as medianfilter
from scipy.signal import medfilt as medianfilter
from matplotlib.figure import Figure as mplfig
import numpy as np
import matplotlib.pyplot as plt
import copy


class Looming_with_background_Analyzer(bc.BaseAnalyzer):
    LoomingWB_version = '1.0.5'#to keep track of class versions, makes easier to check for changes. Starts with 1.0.0 on 18.1.2020
    """
    version 1.0.4 changes timeax such that time coordinate 0  is at the end of looming
    version 1.0.5 overloads plot_stimPhases, because it is a virtual function in baseClass now (in other words, the baseClass function was moved here)
    version 1.0.6 adjustments to plot_stimPhases xlim
    """
    stimulusmap = {0.5:'loom left rot. clw', 1.0:'loom left rot. cck', 1.5:'loom left no rot.',
                  2.0:'loom right rot. clw', 2.5:'loom right rot. cck', 3.0:'loom right no rot.',
                  3.5:'no loom rot. clw', 4.0:'no loom rot. cck'}
    stimulusPhases_inframes = [0,47,60,74] #frame numbers where stuff changes
    stimulus_period_inframes = 60
    preStim = 0.1 #s
    afterStim = 0.25 #s
    
    
    def __init__(self, fpath, *args, **kwargs):
        loader = bc.DataLoader(fpath)
        data, samplingRate = loader.get_data()
        samplingRate = round(samplingRate/100.,0)*100.
        if data.shape[0]==5:
            data[1,:] = medianfilter(data[1,:], 51)
            data[2,:] = medianfilter(data[2,:], 51)
        elif data.shape[0]==4:
            data[1,:] = medianfilter(data[1,:], 9)
            data[2,:] = medianfilter(data[2,:], 9)
        stimBound = loader.get_stimulusBoundaries()
        stimBound[0::2] = stimBound[0::2]-self.preStim*samplingRate
        stimBound[1::2] = stimBound[1::2]+self.afterStim*samplingRate
        metas = loader.get_metadata()
        super(Looming_with_background_Analyzer, self).__init__(stimBound, data, Fs=samplingRate, fname=fpath.split('/')[-1], metadata=metas, **kwargs)
        self.timeax = self.timeax-self.preStim-self.stimulusPhases_inframes[1]*self.meanFramePeriod #center 0 on end of loom
        
    
    def stimulusValidationTests(self):
        #stimIDcrossings = np.array([ seg.boundaries for seg in self.segments]).flatten()
        endAS = np.array([ np.mean(seg.data[self.channelmap['arenaOut'], -10:]) for seg in self.segments ])
        EndFrameRounded = np.round(endAS/5.*self.stimulusPhases_inframes[-1])
        if any(EndFrameRounded<=self.stimulusPhases_inframes[-2]):
            print('ATTENTION: at least one stimulus did not finish the rotating phase')

    def get_stimulusmap(self, idx):
        '''returns a string that is to be used for labels like plot titles
         takes an int that corresponds to a group_index value, and outputs the corresponding label
         The transform from int index to voltage value used as key in the stimulusmap dict is subclass specific'''
        return self.stimulusmap[(idx+1.)*0.5]

    def sum_individual_components(self, combined, candidates):
        f, ax=plt.subplots(2,1)
        # first handle the wingDiff data
        #avgWings = {k: v[self.channelmap['wingDiff']] for k,v in self.meanSegments.items()}
        avgWings = self.get_avg_data('wingDiff')
        ax[0].plot(self.timeax[:avgWings[combined].size], avgWings[combined])
        if len(avgWings[candidates[0]])<len(avgWings[candidates[1]]):
            toplotSum = avgWings[candidates[1]].copy()
            toplotSum[:len(avgWings[candidates[0]])] += avgWings[candidates[0]]
        else:
            toplotSum = avgWings[candidates[0]].copy()
            toplotSum[:len(avgWings[candidates[1]])] += avgWings[candidates[1]]
        [ax[0].plot(self.timeax[:avgWings[pC].size], avgWings[pC], alpha=0.7) for pC in candidates]
        ax[0].plot(self.timeax[:toplotSum.size], toplotSum)
        ax[0].set_xlim([-self.stimulusPhases_inframes[1]*self.meanFramePeriod ,0.45])
        #now the ephys data
        avgEphys = {k: v[self.channelmap['ephys']] for k,v in self.meanSegments.items()}
        ax[1].plot(self.timeax[:avgEphys[combined].size], avgEphys[combined])
        if len(avgEphys[candidates[0]])<len(avgEphys[candidates[1]]):
            toplotSum = avgEphys[candidates[1]].copy()
            toplotSum[:len(avgEphys[candidates[0]])] += avgEphys[candidates[0]]
        else:
            toplotSum = avgEphys[candidates[0]].copy()
            toplotSum[:len(avgEphys[candidates[1]])] += avgEphys[candidates[1]]
        [ax[1].plot(self.timeax[:avgEphys[pC].size], avgEphys[pC], alpha=0.7) for pC in candidates]
        ax[1].plot(self.timeax[:toplotSum.size], toplotSum)
        ax[1].set_xlim([-self.stimulusPhases_inframes[1]*self.meanFramePeriod ,0.45])
        plt.legend([self.stimulusmap[(combined+1)*0.5], self.stimulusmap[(candidates[0]+1)*0.5], self.stimulusmap[(candidates[1]+1)*0.5], 'addition'], fontsize=20)
        self.plot_stimPhases(f)
        return
        
    def plot_stimPhases(self, *args):
        '''adds stimulus phases to plot of averaged response
        if that plot doesn't exist, it is created here '''
        if len(args)==0:
            try:
                self.meanFigure
            except AttributeError:
                self.plotMean()
            ax = self.meanFigure.axes
        else:
            assert( isinstance(args[0], mplfig))
            ax = args[0].axes
        colors = {0:[1.,.34,.13], 1: [.48,.70,.26], 2:[.74,.66,.64]}
        meanFramePeriod = np.median(self.frame_update_periods)
        #demarcations = [0,47,60, 74] #frame numbers where stuff changes
        demarcations = self.stimulusPhases_inframes
        for a in ax:
            y1,y2 = a.get_ylim()
            for i in range(len(demarcations)-1):
                a.fill_between([demarcations[i]*meanFramePeriod+self.timeax[0]+self.preStim, demarcations[i+1]*meanFramePeriod+self.timeax[0]+self.preStim], [y1,y1], [y2,y2], alpha=0.5, color=colors[i])
                #a.fill_between([i,i+1], [y1,y1], [y2,y2], alpha=0.3)
            a.set_ylim([y1,y2])
        return
    
    def separate_flying_resting(self):
        '''split the present Analyzer class into two new classes, one containing the trials where animal is flying, the others where resting
        Assumes that the sorting flying/resting has already happened, using e.g. remove_dud_gui()'''
        saFly = copy.copy(self)
        saRest = copy.copy(self)
        saRest.remove_duds_byList( np.where(saRest.flying)[0])
        saFly.remove_duds_byList( np.where(np.logical_not(saFly.flying))[0])
        return (saFly, saRest)
    
    
    
    def separate_saccading_unreactive(self):
        '''split the present Analyzer class into two new classes, one containing the trials where animal is doing a saccade, the others where not
        Assumes that the sorting saccade/unreactive has already happened, using e.g. remove_dud_gui()'''
        saSaccade = copy.copy(self)
        saStraight = copy.copy(self)
        saStraight.remove_duds_byList( np.where(saStraight.doesSaccade)[0])
        saSaccade.remove_duds_byList( np.where(np.logical_not(saSaccade.doesSaccade))[0])
        return (saSaccade, saStraight)


