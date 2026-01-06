import numpy as np
import torch
import os
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import xarray as xr
import random,re
from wrappers import SRImplicitDownsampled
class MyDataset(dataset):
    def __init__(self, data_dir, start_year, end_year, var_name ='t2m',get_name=False):
        self.start_year = start_year
        self.end_year = end_year
        self.variable = var_name
        self.get_name = get_name
        self.scale_factor = 0.00145877124883646
        self.add_offset = 270.095568832149
        self.file_list = self._get_file_list()
        print(f"Found {len(self.file_list)} files for years {self.start_year}-{self.end_year}")
        self.mean,self.std = self._mean_std()
        print(f"Mean: {self.mean}, Std: {self.std}")
    def _get_file_list(self):
        file_list = []
        for year in range(self.start_year, self.end_year + 1):
            year_dir = os.path.join(data_dir, str(year))
            if not os.path.exists(year_dir):
                continue
            for root,dirs,files in os.walk(year_dir):
                for file in files:
                    if file.endswith('.nc'):
                        file_list.append(os.path.join(root, file))
        return file_list
    def _mean_std(self, sample=False):
        if sample:
            sampled_files = random.sample(self.file_list, min(2000, len(self.file_list)))
        else:
            sampled_files = self.file_list
        data_all = []
        for file in sampled_files:
            ds = xr.open_dataset(file)
            data = ds[self.variable].values
            data = data * self.scale_factor + self.add_offset
            data_all.append(data)
            ds.close()
        data_all = np.array(data_all)
        print('Data shape for mean/std calculation:', data_all.shape)
        mean = np.mean(data_all)
        std = np.std(data_all)
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        ds = xr.open_dataset(file_path)
        data = ds[self.variable].values
        ds.close()
        data = data * self.scale_factor + self.add_offset
        data = torch.from_numpy(data).float()
        data = (data - self.mean) / self.std
        if self.get_name:
            return data, file_path
        else:
            return data


def SR_dataset(data_dir,start_year,end_year,scale_max=8,scale_min=2,inp_size=90,sample_q=5000):
    my_dataset = MyDataset(data_dir, start_year, end_year)
    SR_dataset = SRImplicitDownsampled(dataset=my_dataset, scale_max=scale_max, scale_min=scale_min, inp_size=inp_size, sample_q=sample_q)
    return SR_dataset

if __name__ == "__main__":
    data_dir = '/path/to/your/data'
    dataset = MyDataset(data_dir, start_year=2000, end_year=2020, var_name='t2m')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for batch in dataloader:
        print(batch.shape)
        break