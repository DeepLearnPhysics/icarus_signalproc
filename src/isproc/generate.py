import h5py as h5
import numpy as np
import tqdm

def draw_track(num_tracks_max=5,image_size=128,step_size=0.2,spread=0.5,qmean=80,qspread=0.2,cluster_size=100,min_pixel=5,seed=None):
    '''Generate an image of track-like signal.

    A track-like signal is generated in the following steps:
        1) randomly sample two points = start and end of a track
        2) uniformly (every {step_size} in the pixel unit) sample along the straight line that connects start to the end a
        3) per sampled point along the track, generate {cluster_size} energy depositions. The position of each deposition is sampled
        from the normal distribution with the width {spread} and the amount of energy also sampled from a normal distibution with
        the mean {qmean/cluster_size} and std {qsigma}. 
        4) the generated track is "valid" if the outcome of step 3 filles more than {min_pixel} pixels.

    Parameters:
    -----------
    num_tracks_max : int
        The generated image will contain N tracks where N is sampled between [0,num_tracks_max] (see step 1)
    image_size : int
        The size of a (square) image to be generated. Should be the pixel count on one side.
    step_size : float
        The step-size along the sampled trajectory to generate energy depositions (see step 3)
    spread : float
        The position smearing size in the unit of pixels. This is used as the std of a normal distribution (see step 3)
    qmean : float
        The mean of the sum of all charge depositions per sampled point along the trajectory (note, the sum of all {cluster_size} depositions)
    qspread : float
        The std of all charge depositions per generated energy depositions.
    cluster_size : int
        The number of energy depositions to be sampled per point sampled along the particle trajectory
    min_pixel : int
        The minimum number of pixels in the image for a track to be considered valid
    seed : int
        The numpy random seed for reproducibility. If None, seed is not set.

    Returns:
    --------
    np.ndarray
        An image of tracks with the shape (image_size,image_size)

    '''
    if not seed is None:
        np.random.seed(int(seed))
    img=np.zeros(shape=(image_size,image_size),dtype=np.float32)
    #window_size = np.sqrt(2)*image_size
    window_size = image_size
    coord_min=(window_size - image_size)/2.
    coord_max=window_size-coord_min

    num_tracks = int(np.random.random()*num_tracks_max)+1
    while num_tracks > 0:
        
        xs,ys,xe,ye=np.random.random(4) * window_size-coord_min
            
        total_length = np.sqrt((ye-ys)**2 + (xe-xs)**2)
        unit_dir = np.array([ye-ys,xe-xs]) / total_length
        num_points = int(total_length / step_size)+1
        
        charge=np.random.uniform(qmean/cluster_size,qspread,num_points*cluster_size).astype(np.float32)
        xidx=np.random.uniform(0.,spread,num_points*cluster_size).astype(np.float32)
        yidx=np.random.uniform(0.,spread,num_points*cluster_size).astype(np.float32)
        for i in range(num_points):
            start=i*cluster_size
            end=(i+1)*cluster_size
            xidx[start:end] += i*step_size*unit_dir[0]+xs
            yidx[start:end] += i*step_size*unit_dir[1]+ys

        mask = (xidx > coord_min) & (xidx < coord_max) & (yidx > coord_min) & (yidx < coord_max)

        if mask.sum() < min_pixel:
            continue
        #else:
        #    print(f'accept ({xidx.min(),yidx.min()}) => ({yidx.min(),yidx.max()})')

        num_tracks -= 1
        
        xidx = (xidx[mask]-coord_min).astype(int)
        yidx = (yidx[mask]-coord_min).astype(int)
        charge = charge[mask]

        img[xidx,yidx]+=charge        
            
    return img

def update_signal_in_h5(fname,signal_key,**kwargs):
    '''Given an existing h5 file, replace the track signal data array with a generated one

    This function is used to replace the signal data definition in an existing data file.
    The existing signal data product is used to set the total image statistics to be generated.

    Parameters:
    -----------
    fname : string
        The path to a HDF5 file
    signal_key : string
        The attribute name of the signal array within the HDF5 file.
    kwargs : dict
        The parameters passed down to the generator (draw_track) function.

    '''
    data=dict()
    signal_stat=0
    with h5.File(fname,'r') as f:
        if not signal_key in [str(key) for key in f.keys()]:
            raise KeyError(f'"Signal" key "{signal_key}" does not exist in the file {fname}')
        for key in f.keys():
            if key == signal_key:
                signal_stat = f[signal_key].shape[0]
                continue
            data[key]=np.array(f[key])

    signal_data = []
    for _ in tqdm.tqdm(range(signal_stat)):
        signal_data.append(draw_track(**kwargs).flatten())
    data[signal_key]=np.stack(signal_data)

    fname=fname+'.updated'
    print('Saving data to',fname)
    with h5.File(fname,'w') as f:
        for key,val in data.items():
            print('  ... writing',key)
            f.create_dataset(key,data=val)
    print('Done saving')

def add_signal_to_h5(fname,signal_key,signal_stat,**kwargs):
    '''Given an existing h5 file, add a newly generated track signal data array 

    This function is used to add the signal track data definition in an existing data file

    Parameters:
    -----------
    fname : string
        The path to a HDF5 file
    signal_key : string
        The attribute name of the signal array within the HDF5 file.
    signal_stat : int
        The number of signal track images to be generated.
    kwargs : dict
        The parameters passed down to the generator (draw_track) function.

    '''
    data=dict()
    with h5.File(fname,'r') as f:
        if signal_key in [str(key) for key in f.keys()]:
            raise KeyError(f'"Signal" key "{signal_key}" already exist in the file {fname}')

    signal_data = []
    for _ in tqdm.tqdm(range(signal_stat)):
        signal_data.append(draw_track(**kwargs).flatten())
    data[signal_key]=np.stack(signal_data)

    print('Saving data to',fname)
    with h5.File(fname,'a') as f:
        for key,val in data.items():
            print('  ... writing',key)
            f.create_dataset(key,data=val)
    print('Done saving')