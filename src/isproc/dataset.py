"""Contains dataset classes to be used by the model."""
import numpy as np
import torch
from torch.utils.data import Dataset
import time, os, h5py

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGOverlay(Dataset):

    def __init__(self, files=[], dtype=torch.float32, in_memory=False, ignore=[], device=DEFAULT_DEVICE):
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
        self._device= DEFAULT_DEVICE
        if files:
            self.register_files(files,ignore)

    def __del__(self):

        if hasattr(self,"_h5fs") and self._h5fs:
            for f in self._h5fs:
                f.close()

    def register_files(self,files,ignore=[]):
        """Register the input files. If self.in_memory, also copy data into memory.

        Parameters
        ----------
        files : str or list
            Input files (one file if string, multiple files if list)
        ignore : str or list
            Data attributes in some or all input files to be ignored from reading
        """
        t0=time.time()
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

            print('[SGOverlay] Reading file %-2d: %s' % (idx,file_name))

            # Check if data attribute keys found exist in all files
            num_entry = None
            for k in keys:
                if not k in f.keys():
                    raise KeyError(f'[SGOverlay] Attribute "{k}" not found in {file_name}')
                if num_entry is None:
                    num_entry = f[k].shape[0]
                elif not num_entry == f[k].shape[0]:
                    raise ValueError(f'Attribute {k} has length {f[k].shape[0]} but expected {num_entry} from {keys[0]}')

                if self._in_memory:
                    in_memory[k].append(torch.as_tensor(np.array(f[k]),dtype=self._dtype))

            num_entries.append(num_entry)
            file_index.append(np.ones(num_entry,dtype=np.uint16))
            file_index[-1] *= idx

            self._h5fs.append(f)

        if self._in_memory:
            print('[SGOverlay] Loading data in memory')
            self._in_memory=dict()
            for key,val in in_memory.items():
                self._in_memory[key] = torch.cat(val).to(self._device)


        self._keys = keys
        self._files = files
        self._num_entries = np.array(num_entries)
        self._file_index = np.concatenate(file_index)
        print('[SGOverlay] Finished reading files (%.3f [s] %d entries)' % (time.time()-t0,self._num_entries.sum()))


    def to(self,device):
        if self._in_memory:
            self._in_memory = self._in_memory.to(device)
        self._device = device


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
            return {key:self._in_memory[key][idx] for key in self._keys}
        else:
            data={key:[] for key in self._keys}
            file_index = self._file_index[idx]
            for fidx in np.unique(file_index):
                offset = self._num_entries[:fidx+1].sum()
                local_idx = idx - offset
                for key in self._keys:
                    data[key].append(torch.as_tensor(self._h5fs[fidx][key][local_idx],dtype=self._dtype))
            if len(data[self._keys[0]]) < 2:
                for key in self._keys:
                    data[key]=data[key][0]
            else:
                for key in self._keys:
                    data[key]=torch.cat(data[key]).to(self._device)
            return data


    def data_keys(self):
        """Returns a list of data product names.

        Returns
        -------
        List[str]
            List of data product names
        """
        return self._keys




