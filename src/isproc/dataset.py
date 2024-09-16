"""Contains dataset classes to be used by the model."""
import numpy as np
from torch.utils.data import Dataset
import time, os, h5py

class SGOverlay(Dataset):

    def __init__(self, files=[], dtype=np.float32, in_memory=False, ignore=[], read_fraction=[0.0,1.0]):
        """Instantiates the SGOverlay dataset.

        Parameters
        ----------
        files : str or list
            Input files (one file if string, multiple files if list)
        dtype : str
            Data type to cast the input data to (to match the downstream model)
        in_memory : bool
            If True, data from all input files are read and stored in CPU RAM to minimize file access
        ignore : str or list
            The name (string) of data attributes to ignore upon reading the input data file contents
        """

        self._in_memory = in_memory
        self._dtype = dtype
        if files:
            self.register_files(files,ignore,read_fraction)

    def __del__(self):

        if hasattr(self,"_h5fs") and self._h5fs:
            for f in self._h5fs:
                f.close()

    def register_files(self,files,ignore=[],read_fraction=[0.0,1.0],image_shape=[128,128]):
        """Register the input files. If self.in_memory, also copy data into memory.

        Parameters
        ----------
        files : str or list
            Input files (one file if string, multiple files if list)
        ignore : str or list
            Data attributes in some or all input files to be ignored from reading
        """
        t0=time.time()
        self._image_shape=list(image_shape)
        # constrain the files type
        if isinstance(files, str):
            files=[files]
        if not isinstance(files,list):
            raise TypeError('[SGOverlay] The constructor argument "files" must be str or list')
        elif len(files)<1:
            print('[SGOverlay] The file list is empty')

        # constrain the ignore type
        if isinstance(ignore, str):
            ignore=[ignore]
        if not isinstance(ignore,list):
            raise TypeError('[SGOverlay] The constructor argument "ignore" must be str or list')

        # ensure the input files are present/valid
        for f in files:
            if not os.path.isfile(f):
                FileNotFoundError(f'[SGOverlay] Input file not found: {f}')

        self._h5fs = []
        in_memory   = {}
        file_index  = []
        local_index = []
        num_entries = []
        keys = []
        #
        # Loop over files, open and read
        #
        for idx,file_name in enumerate(files):

            f = h5py.File(file_name,'r')

            # First file: found all data attribute keys
            if len(keys) < 1:
                keys = [str(k) for k in f.keys() if not k in ignore]
                print('[SGOverlay] Reading data attributes:',keys)

                if self._in_memory:
                    in_memory = {key:[] for key in keys}

            print('[SGOverlay] Reading file %-2d: %s ... %.2f => %.2f' % (idx,file_name,read_fraction[0],read_fraction[1]))

            # Check if data attribute keys found exist in all files
            num_entry = None
            start,end=read_fraction
            for k in keys:
                if not k in f.keys():
                    raise KeyError(f'[SGOverlay] Attribute "{k}" not found in {file_name}')
                if num_entry is None:
                    num_entry = f[k].shape[0]
                    start = int(start*num_entry)
                    end   = int(end*num_entry)
                elif not num_entry == f[k].shape[0]:
                    raise ValueError(f'Attribute {k} has length {f[k].shape[0]} but expected {num_entry} from {keys[0]}')

                if self._in_memory:
                    in_memory[k].append(np.array(f[k][start:end],dtype=self._dtype))
                    shape = [end-start,1]+list(self._image_shape)
                    in_memory[k][-1]=in_memory[k][-1].reshape(shape)
            num_entry = end-start
            num_entries.append(num_entry)
            local_index.append(np.arange(start,end))
            file_index.append(np.ones(num_entry,dtype=np.uint16))
            file_index[-1] *= idx

            self._h5fs.append(f)

        if self._in_memory:
            print('[SGOverlay] Loading data in memory')
            self._in_memory=dict()
            for key,val in in_memory.items():
                self._in_memory[key] = np.concatenate(val)

        self._keys = keys
        self._files = files
        self._num_entries = np.array(num_entries)
        file_index  = np.concatenate(file_index )
        local_index = np.concatenate(local_index)

        self._file_index,self._local_index=dict(),dict()
        for key in self._keys:
            self._file_index[key]  = file_index.copy()
            self._local_index[key] = local_index.copy()
        self._raw_file_index  = file_index 
        self._raw_local_index = local_index

        # index mappings
        self._forward_mapping, self._backward_mapping = dict(), dict()
        for key in self._keys:
            self._forward_mapping[key]  = np.arange(len(self))
            self._backward_mapping[key] = np.arange(len(self))

        print('[SGOverlay] Finished reading files (%.3f [s] %d entries)' % (time.time()-t0,self._num_entries.sum()))

    def __len__(self):
        """Returns the lenght of the dataset (in number of batches).

        Returns
        -------
        int
            Number of entries in the dataset
        """
        return self._num_entries.sum()

    def __getitem__(self, idx):
        """Returns one element of the dataset.

        Parameters
        ----------
        idx : int or array of int
            Index (or fancy indicies) of the dataset to load

        Returns
        -------
        dict
            Dictionary of data product names and their associated data
        """
        # Read in a specific entry

        if self._in_memory:
            data = {key:self._in_memory[key][idx] for key in self._keys}
            for key in self._keys:
                data['index_'+key] = self._forward_mapping[key][idx]
            return data
        else:
            data={key:[] for key in self._keys}

            if hasattr(idx,"__len__"):
                for key in self._keys:
                    file_index  = self._file_index[key][idx]
                    local_index = self._local_index[key][idx]
                    data[key]=np.array(self._h5fs[file_index][[key]*len(file_index)][local_index],dtype=self._dtype)
                    shape=[len(idx)]+self._image_shape
                    data[key].reshape(shape)
                    data['index_'+key] = self._forward_mapping[key][idx]
            else:
                for key in self._keys:
                    file_index  = self._file_index[key][idx]
                    local_index = self._local_index[key][idx]
                    data[key]=self._h5fs[file_index][key][local_index]
                    data[key].reshape(self._image_shape)
                    data['index_'+key] = self._forward_mapping[key][idx]
            return data

    def shuffle(self,keep_alignment=False):

        # first, align back to the raw data
        self.reset_shuffle()

        if keep_alignment:
            idx_v = np.arange(len(self))
            np.random.shuffle(idx_v)
            for key in self._keys:
                self._forward_mapping[key]  = idx_v
                self._backward_mapping[key] = np.zeros(len(idx_v),dtype=int)
                self._backward_mapping[key][idx_v] = np.arange(len(idx_v))
                self._file_index[key]  = self._file_index[key][idx_v]
                self._local_index[key] = self._local_index[key][idx_v]
                self._in_memory[key]   = self._in_memory[key][idx_v]
        else:
            for key in self._keys:
                idx_v = np.arange(len(self))
                np.random.shuffle(idx_v)
                self._forward_mapping[key]  = idx_v
                self._backward_mapping[key] = np.zeros(len(idx_v),dtype=int)
                self._backward_mapping[key][idx_v] = np.arange(len(idx_v))
                self._file_index[key]  = self._file_index[key][idx_v]
                self._local_index[key] = self._local_index[key][idx_v]
                self._in_memory[key]   = self._in_memory[key][idx_v]

    def reset_shuffle(self):
        for key in self._keys:
            self._file_index[key]  = self._file_index[key][self._backward_mapping[key]]
            self._local_index[key] = self._local_index[key][self._backward_mapping[key]]
            assert (self._file_index[key]  == self._raw_file_index).sum()  == len(self._file_index[key])
            assert (self._local_index[key] == self._raw_local_index).sum() == len(self._local_index[key])
            self._in_memory[key] = self._in_memory[key][self._backward_mapping[key]]

            self._forward_mapping[key] = np.arange(len(self._forward_mapping[key]))
            self._backward_mapping[key] = np.arange(len(self._backward_mapping[key]))

    def data_keys(self):
        """Returns a list of data product names.

        Returns
        -------
        List[str]
            List of data product names
        """
        return self._keys




