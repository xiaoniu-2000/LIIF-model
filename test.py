import os
import LIIF
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np
from MyDataset import MyDataset
import LIIF
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

if __name__ == '__main__':
    #超参数
    batch_size = 64
    scale_factors = [2, 4, 8]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for scale_factor in scale_factors:
        data_dir = '/path/to/your/data'
        save_dir = '/path/to/save/results'
        os.makedirs(save_dir, exist_ok=True)
        hidden_list = [256, 256, 256, 256]
        model = LIIF.LIIF(1, hidden_list).to(device)
        test_dataset = MyDataset(data_dir, start_year=2000, end_year=2020, var_name='t2m')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load(f'./checkpoints/liif_t2m_scale{scale_factor}.pth', map_location=device))
        model.eval()
        batch_for_coord = next(iter(test_loader))
        img_shape = [int(batch_for_coord.shape[-2]), int(batch_for_coord.shape[-1])]
        coord = LIIF.make_coord(img_shape)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / img_shape[0]
        cell[:, 1] *= 2 / img_shape[1]

        lr_shape = [math.floor(img_shape[0] / scale_factor), math.floor(img_shape[1] / scale_factor)]
        for img_hr, filenames in test_loader:
            img_hr = img_hr.to(device)
            img_lr = F.interpolate(img_hr.unsqueeze(1), size=lr_shape, mode='bilinear', align_corners=False).squeeze(1)
            img_lr = img_lr.to(device)
            pred = batched_predict(model, img_lr, coord.unsqueeze(0).to(device), cell.unsqueeze(0).to(device), bsize=30000)
            pred = pred.detach().cpu().numpy()
            for i,filename in enumerate(filenames):
                basename = os.path.basename(filename)
                output_path = os.path.join(save_dir, basename)
                real_output = pred[i] * test_dataset.std.numpy() + test_dataset.mean.numpy()
                np.save(output_path, real_output)
                print(f'File {basename} saved to {output_path}s')
