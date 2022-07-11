""" Collection of function or classes for data analysis that go beyond the scope of the ndividual Analyzer classes.
Either they are not specific to a certain type of experiment, but don't provide functionality necessary for all use cases.
Or ?
Anyway, at this point the BaseAnalyzer class gets pretty bloated, even dealing with some data in/output as well. So we will put this current one away somewhere.
If it doesn't make sense after all, we can still integrate it into other classes.
"""

import numpy as np
from scipy import signal
import pdb
import h5py


def straightLineFiltering(y, samplesPerFrame, windowFactor=1.4, postEnhancement=False, pruneXsteps=False):
    """ VERSION OF "28.1.2021 """
    # in this window take mean of first N values and last N values as y-values of both frames
    #   first get all y-values of the steps
    samplesPerFrame = int(samplesPerFrame)
    windowSize = int(windowFactor*samplesPerFrame)
    averagingWindow = int(windowSize/5)
    overlap = int(0.3*samplesPerFrame)
    i=overlap
    j=0
    ysteps = np.zeros(int(y.size/samplesPerFrame)+2)
    ysteps[0]=np.mean(y[:averagingWindow])
    while i<y.size:
        chunk=y[i-overlap:i+windowSize-overlap]
        ysteps[j+1]=np.median(chunk[-averagingWindow:])
        i+=samplesPerFrame
        j+=1
    ysteps[-1]=np.mean(y[-averagingWindow:])
    # fit this y-Step to the data with variable x ; x is bounded to a range of kinefly frame length +-10%
    #   start with greatest step instead of first
    xsteps=np.zeros(ysteps.size+1, dtype=np.int)
    whereStart=np.argmax(np.abs(np.diff(ysteps)))+1

    #  first step needs more degrees of freedom
    xrange=np.arange(0,int(2*samplesPerFrame), dtype=np.int) #larger range
    chunkIdx = (whereStart-1)*samplesPerFrame+xrange
    # limit to where indices are valid (take care of overflowing left/right)
    validChunkIdx = np.logical_and(chunkIdx>0, chunkIdx<y.size)
    xrange = xrange[validChunkIdx]
    chunk = y[chunkIdx[validChunkIdx]]
    #do least squares estimate
    ssqErr= [np.sum((np.concatenate((np.full(xstep, ysteps[whereStart-1]), np.full(xrange.size-xstep, ysteps[whereStart])))-chunk)**2) for xstep in xrange]
    whereMin = np.argmin(ssqErr)
    xsteps[whereStart] = (whereStart-1)*samplesPerFrame+xrange[whereMin]
    # ready to loop through all steps 
    #first backward
    fringe = np.arange(whereStart-1,0,-1, dtype=np.int)
    xrange= np.arange(int(samplesPerFrame*0.9), int(samplesPerFrame*1.1), +1, dtype=np.int)
    xprev =xsteps[whereStart] # keep track of last frame fit for probable localization of x
    chunkLims = np.array([samplesPerFrame/2,samplesPerFrame/2*3], dtype=np.int)
    for f in fringe:
        chunk = y[xprev-chunkLims[1]:xprev-chunkLims[0]]
        if xprev-chunkLims[1]<0:
            chunk = y[:xprev-chunkLims[0]]
            if xprev-chunkLims[0] <=0:
                chunkLims[0]=1
                chunk = y[:yprev-chunkLims]
        ssqErr= [np.sum((np.concatenate((np.full(chunk.size+chunkLims[0]-xst, ysteps[f-1]), np.full(xst-chunkLims[0], ysteps[f])))-chunk)**2) for xst in xrange]
        whereMin = np.argmin(ssqErr)
        if not xsteps[f]==0:
            raise Exception("Unexpected f index")
        xsteps[f]=xprev-xrange[whereMin]
        if xsteps[f]<0:
            raise Exception("invalid, index too low")
        xprev=xsteps[f]

    #then forward
    fringe = np.arange(whereStart+1,xsteps.size-2)
    xrange= np.arange(int(samplesPerFrame*0.85), int(samplesPerFrame*1.15), +1, dtype=np.int) #smaller possible range
    xprev =xsteps[whereStart]
    for f in fringe:
        chunk = y[xprev+chunkLims[0]:xprev+chunkLims[1]]
        ssqErr= [np.sum((np.concatenate((np.full(xst-chunkLims[0], ysteps[f-1]), 
                                         np.full(chunkLims[0]+chunk.size-xst, ysteps[f])))-chunk)**2) 
                 for xst in xrange[(chunkLims[0]+chunk.size-xrange)>0] ] #index to avoid neg. dim. in np.full
        if len(ssqErr)==0: #simple hack, not totally wrong either, mostly hints at broken data assumptions
            xsteps[f]=xprev
        else:
            whereMin = np.argmin(ssqErr)
            if not xsteps[f]==0:
                raise Exception("Unexpected f index")
            xsteps[f] = xprev+xrange[whereMin]
            if xsteps[f]>y.size:
                raise Exception("invalid, index too large")
        xprev=xsteps[f]
    #last one manually
    if not whereStart==ysteps.size-1: #in which case it's all done already
        f=ysteps.size-1
        chunk=y[xprev+chunkLims[0]:]
        xrange=np.arange(chunkLims[0],chunkLims[0]+chunk.size)
        if xprev+chunkLims[0] >= y.size:
            chunkLims[0] = 0
            chunk = y[xprev:]
            xrange = np.arange(0,chunk.size)
        ssqErr= [np.sum((np.concatenate((np.full(xst-chunkLims[0], ysteps[f-1]),
                                         np.full(chunkLims[0]+chunk.size-xst, ysteps[f])))-chunk)**2)
                 for xst in xrange[(chunkLims[0]+chunk.size-xrange)>=0] ]
        whereMin = np.argmin(ssqErr)
        xsteps[f]=xprev+xrange[whereMin]

    xsteps[0] = 0
    xsteps[-1] = y.size
    
    if postEnhancement:
        for i in range(ysteps.size):
            ysteps[i] = np.median(y[xsteps[i]:xsteps[i+1]])
        if int(sum(np.isnan(ysteps)))==1: #not sure why this happens though
            idx = np.where(np.isnan(ysteps))[0]
            ysteps[idx] = (ysteps[idx-1]+ysteps[idx+1])/2.
    
    if pruneXsteps:
        nullSteps = np.diff(xsteps)<=1
        xsteps = np.concatenate((xsteps[:1], xsteps[1:][np.logical_not(nullSteps)] ))
        ysteps = ysteps[np.logical_not(nullSteps)]
    
    yfit = np.concatenate([np.full(dx,ysteps[i]) for i,dx in enumerate(np.diff(xsteps))])
        
    # maybe check the final global functional to decide if it's ok or readjust some more
    #f,ax=plt.subplots(1,1)
    #ax.plot(y)
    #ax.plot(yfit)
    #ax.plot( np.concatenate((np.full(xsteps[1],ysteps[0]), np.concatenate([np.full(samplesPerFrame, ys) for ys in ysteps[1:] ]) )))
    #ax.legend(['original','adjusted x-steps','rigid x-steps'])
    
    return yfit, xsteps,ysteps


def resampleAnalyzer(anlyz, targetFs=10000.):
    """ VERSION OF 2.12.2020 end of day """
    #isnecessary = not ( np.all( abs(np.array([seg.samplingRate for seg in anlyz.segments])-15000.)<10) or np.all(abs(np.array([seg.samplingRate for seg in anlyz.segments])-targetFs)<10) )
    isnecessary = True
    if not isnecessary:
        raise Warning('resampling not necessary, skipped')
        return
    maxLen = 0
    for seg in anlyz.segments:
        proportion = targetFs/seg.samplingRate
        if abs(proportion-1.)<0.005: #irrelevant, conversion from Matlab to Python warped the rate a little
            continue
        seg.data = signal.resample(seg.data, int(seg.data.shape[1]*proportion), axis=1)
        seg.samplingRate = targetFs
        if seg.data.shape[1] > maxLen:
            maxLen = seg.data.shape[1]
    anlyz.samplingRate = targetFs
    return
    

def fitSteps(segIdx):
    #f,ax=plt.subplots(1,1)
    #fitData = saFly.meanSegments[2][3]
    fitData = saFly.segments[segIdx].data[3]
    estimate = np.zeros(fitData.size)
    nSteps = saFly.stimulusPhases_inframes[-1]
    meanYstep = 5./(nSteps+1) #prior knowledge about y steps
    yFitRange = np.arange(meanYstep-0.025,meanYstep+0.025,0.005)
    #get prior about expected x size
    #arenaResetIdx = np.where(np.abs(np.diff(saFly.meanSegments[0][-3]))>2)[0]
    #meanXstep = (arenaResetIdx[1]-arenaResetIdx[0])//64
    arenaResetIdx = int(np.mean(np.concatenate([np.where(np.abs(np.diff(seg.data[saFly.channelmap['arenaOut']]))>2)[0] for seg in saFly.segments])))
    meanXstep = (arenaResetIdx-saFly.preStim*saFly.samplingRate)//nSteps
    #first round of fitting
    #ax.plot(fitData)
    xSteps = np.zeros(nSteps, dtype=int)
    for s in range(nSteps):
        #yFitRange = meanYstep*np.arange(0.5,1.6,0.1)
        xFitRange = arenaResetIdx-np.arange(meanXstep*(nSteps-s-1), meanXstep*(nSteps-s+1),2, dtype=np.int)
        res=-np.ones((xFitRange.size))
        for j,thisX in enumerate(xFitRange):
            estimateNew = np.concatenate((estimate[:thisX],estimate[thisX:]+meanYstep))
            res[j]= np.sum((fitData-estimateNew)**2)
        bestIdx = np.argmin(res)
        xSteps[s] = int(xFitRange[bestIdx])
        #don't #fit best Y step
        #bestY = np.argmin([np.sum((fitData- np.concatenate((estimate[:xFitRange[bestIdx]],estimate[xFitRange[bestIdx]:]+thisY)))**2) for thisY in yFitRange])
        estimate = np.concatenate((estimate[:xFitRange[bestIdx]],estimate[xFitRange[bestIdx]:]+meanYstep))
    #ax.plot(estimate)
    return xSteps


class PatternReader(dict):
    patternKeys = ['x_num', 'y_num', 'num_panels', 'gs_val', 'row_compression', 'Pats', 'Panel_map', 'BitMapIndex', 'data']
    
    #from numpy import ndarray
    
    def __init__(self, path):
        try:
            import scipy.io as scio
            self.patternfile = scio.loadmat(path)
            pattern = self.patternfile['pattern']
            pattern = self.unravel(pattern)
            super(PatternReader, self).__init__(zip(self.patternKeys, pattern))
        except NotImplementedError:
            self.patternfile = h5py.File(path, 'r')
            pattern = self.resolveHDF(self.patternfile['pattern'])
            super(PatternReader, self).__init__( pattern.items() )
        
        for k,v in self.items():
            self[k] = self.unravel(v)
        #invert along y axis because arena works top-to-bottom, not bottom-to-top (y=0 is top line of LEDs)
        self['Pats'] = self['Pats'][::-1,...]
        self.post_process()
        return
    
    def post_process(self):
        # expand to full pattern size
        if self['row_compression']==1:
            assert(self['Pats'].shape[0] == self['Panel_map'].shape[0])
            fullpattern = np.repeat(self['Pats'], 8, axis=0) # ->8x8 pixels per panel
        else:
            fullpattern = self['Pats']
        # correct the full pattern if it's larger than arena
        if self['Panel_map'].shape[1]>11:
            fullpattern = fullpattern[:, -11*8:]        
        # add extra dimension if y-num was 1, so that all patterns have same dimensionality
        if self['y_num']==1:
            assert(self['Pats'].ndim==3)
            fullpattern = fullpattern[...,None] # adding new dimension at end (ellipsis syntax)
        fullpattern = fullpattern.astype('float')
        fullpattern /=np.nanmax(fullpattern)
       
        self['fullpattern'] = fullpattern
        return
    
    @staticmethod
    def addStationaryPeriod(pattern, preStim, afterStim, framerate):
        beforePiece = np.tile(pattern[:,:, 0:1 ,:], [1,1,int(preStim*framerate+0.5),1])
        afterPiece = np.tile(pattern[:,:, -1: ,:], [1,1,int(afterStim*framerate+0.5),1])
        pattern = np.concatenate((beforePiece, pattern, afterPiece),axis=2)
        return pattern
    
    def recreateStimulus(self, duration_s, framePeriod):
        fullpattern = self['fullpattern']
        n_frames = int(duration_s/framePeriod+0.5)
        patternLen = fullpattern.shape[2]
        # tile along time axis and append piece that did not run all the way
        fullpattern = np.tile(fullpattern, [1,1,n_frames//patternLen,1])
        fullpattern = np.concatenate((fullpattern, self['fullpattern'][:,:, :n_frames%patternLen ,:]), axis=2)
        self['prolongedPattern'] = fullpattern
        return 
    
    def unravel(self, arr):
        if not isinstance(arr, (list,tuple,np.ndarray, )):
            return arr 
        if len(arr)>1:
            return arr
        elif len(arr)==0:
            return []
        else: # same as elif len(arr)==1:
            return self.unravel(arr[0])
        
    def resolveHDF(self, hdfGroup):
        d = {}
        for name, content in hdfGroup.items():
            if isinstance(content, h5py.Group):
                d[name] = self.resolveHDF(content)
            elif isinstance(content, h5py.Dataset):
                d[name] = np.array(content).T
            else:
                d[name] = content
        return d
        
    def showAnimation(self):
        #make colormap
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        # define black/green colormap
        newcm = np.array([[0.,0.,0.,1.], [0.,.5,0.,1.] , [0.,.9,0.,1.]])
        newcm = ListedColormap(newcm)
        #make animated figure
        fullpattern = self['fullpattern']
        f,ax=plt.subplots(1,1)
        qm = ax.pcolormesh(fullpattern[:,:,0,0], cmap=newcm)
        qm.animated = True
        for i in range(fullpattern.shape[2]):
            qm.set_array(fullpattern[:,:,i%fullpattern.shape[2],0].ravel())
            plt.draw()
            plt.pause(0.05)
        return
        
    @staticmethod
    def calculateOpticFlow(fullpattern, stimID):
        """legacy, strongly discouraged to use anymore because it is usually nonsense"""
        dEdt = np.diff(np.squeeze(fullpattern[:,:,:,stimID]),axis=2)
        dEdx = np.diff(np.squeeze(fullpattern[:,:,:,stimID]),axis=1)
        dEdy = np.diff(np.squeeze(fullpattern[:,:,:,stimID]),axis=0)#this should be 0
        u = dEdt[:,1:,:]/dEdx[:,:,:-1]
        v = dEdt[1:,:,:]/dEdy[:,:,:-1]
        u[np.isnan(u)] = 0.
        u[np.isinf(u)] = 0.
        v[np.isnan(v)] = 0.
        v[np.isinf(v)] = 0.
        return u,v

    @staticmethod
    #make use of what I know about stimulus
    def calcOptflow_dirTuning(pat, stimID):
        """Meant to calculate the optic flow of a pattern pat according to the a-priori 
        knowledge we have about the direction tuning stimulus protocol
        input: pat  as full-size arena pattern with 4 dimensions
        returns ndarrays u and v for horizontal and vertical optic flow respectively, ndim is same as input, shape[1] and shape[2] reduced by 1"""
        dEdt = np.diff(pat[...,stimID],axis=2)
        #dEdx = np.diff(pat['fullpattern'],axis=1)
        divisor = pat[:,:-1,:-1,stimID]
        #u = -dEdt[:,1:,:,yid]/dEdx[:,:,:-1,yid]
        u = dEdt[:,1:,:]/divisor
        u[np.isnan(u)]=0.
        u[np.isinf(u)]=0.
        v = np.zeros_like(u)
        return u,v
    
    @staticmethod
    def calcOptflow_RFHS(pat, stimID):
        """Meant to calculate the optic flow of a pattern pat according to the a-priori 
        knowledge we have about the horizontal moving stripe stimulus protocol
        input: pat  as full-size arena pattern with 4 dimensions
        returns ndarrays u and v for horizontal and vertical optic flow respectively, ndim is same as input, shape[1] and shape[2] reduced by 1"""
        dEdt = np.diff(pat[...,stimID],axis=2)
        #dEdx = np.diff(pat['fullpattern'],axis=1)
        divisor = pat[:,:-1,:-1,stimID]
        #u = -dEdt[:,1:,:,yid]/dEdx[:,:,:-1,yid]
        u = dEdt[:,1:,:]/divisor
        u[np.isnan(u)]=0.
        u[np.isinf(u)]=0.
        v = np.zeros_like(u)
        return u,v
        
    @staticmethod
    def calcOptflow_loomingWB(pat, stimID, phaseLimit, **kwargs):
        """Calculates optic flow matrices for all arena pixels according to what we know about pattern. 
        This includes detecting the center of the expanding circle unless given in kwargs as "loomCenter"
        input: pattern,   stimulus ID for which to calculate , and phaseLimits as single int where to 
        switch from expanding disc to optomotor mode of calculation"""
        centerDefinite = kwargs.get('center',None)
        if centerDefinite is None:
            allcenter=[]
            for r in [3,5,9,12]:
                allcorrs = np.ones(pat.shape[:2])
                for t in range(0, phaseLimit):
                    #expected radius
                    #r = round(-np.tan(-speed*t*anlyz.meanFramePeriod/2)*32)
                    #print(r)
                    r=5
                    mask = np.meshgrid(np.arange(-r-1,r+1.1), np.arange(-r-1,r+1.1))
                    mask = ((mask[0]**2+mask[1]**2) <= r**2 ).astype(float)
                    cr = signal.correlate2d(1-pat[:,:,t,stimID], mask, 'same' )
                    cr[cr<0.5*cr.max()]=0
                    allcorrs = allcorrs*cr
                    #print(center, p2)
                centerCorrel = np.unravel_index(np.argmax(allcorrs),pat.shape[:2])
                allcenter.append(centerCorrel)
            allcenter = np.array(allcenter)
            centerDefinite=np.mean(allcenter,axis=0)
        #now that we know the center, get to the actual action
        dt = np.diff(pat[...,stimID], axis=2)
        u = np.zeros_like(dt)
        v = np.zeros_like(dt)
        x,y = np.meshgrid(np.arange(88),np.arange(32))
        r = np.sqrt((x-centerDefinite[0])**2.+(y-centerDefinite[1])**2)
        u[:,:,:phaseLimit] = -dt[:,:,:phaseLimit]* ((x[:,:]-centerDefinite[0])/r[:,:])[...,None] #x/r is cosine
        v[:,:,:phaseLimit] = -dt[:,:,:phaseLimit]* ((y[:,:]-centerDefinite[1])/r[:,:])[...,None] #y/r is sine
        u = u[:,1:]
        v = v[:,1:]
        
        uprime,vprime = PatternReader.calcOptflow_RFHS(pat[:,:,phaseLimit:,:], stimID)
        u[:,:,phaseLimit:] = uprime
        v[:,:,phaseLimit:] = vprime
        u[np.isnan(u)] = 0.
        v[np.isnan(v)] = 0.        
        return u,v
