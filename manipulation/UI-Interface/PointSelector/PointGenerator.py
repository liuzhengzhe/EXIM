from Model.config import opts
from Model.model import Model
import numpy as np
import torch

class PointGenerator(object):
    def __init__(self):

        ## define model
        opts.pretrain_model_G = "models_final/600_animal-pose_G.pth"
        # opts.pretrain_model_G = "Chair_v1_300.pth"
        opts.log_dir = "log/20210113-2128"
        self.model = Model(opts)
        self.model.build_model_eval()
        self.load_model()
        self.model.ball_path = 'template/2048.xyz'
        self.model.opts.np = 2048

    def generate_pointclouds(self, cnt : int):

        ## reading ball
        ball = self.model.read_ball()
        x = np.expand_dims(ball, axis=0)
        x = np.tile(x, (cnt, 1, 1))
        x = torch.from_numpy(x).float().cpu()

        ### get noise
        points_noise = np.random.normal(0, 0.2, (cnt, 1, self.model.opts.nz))
        points_noise = np.tile(points_noise, (1, self.model.opts.np, 1))

        ## generate the pointscloud
        z = torch.from_numpy(points_noise).float().cpu()

        points_arr = self.model.G(x, z).transpose(1, 2)
        points_arr = points_arr.detach().cpu().numpy()
        return points_arr, points_noise

    def generate_points_from_noise(self, noise):
        ## reading ball
        ball = self.model.read_ball()
        x = np.expand_dims(ball, axis=0)
        x = torch.from_numpy(x).float().cpu()

        ## reading the
        noise = np.expand_dims(noise, axis=0)
        z = torch.from_numpy(noise).float().cpu()

        points_arr = self.model.G(x, z).transpose(1, 2)
        points_arr = points_arr.detach().cpu().numpy()

        return points_arr

    def interpolation_noise(self, z1, z2, selection, alpha, use_latent = False):
        ## reading ball
        ball = self.model.read_ball()
        x = np.expand_dims(ball, axis=0)
        x = torch.from_numpy(x).float().cpu()

        ## noise
        z1 = np.expand_dims(z1, axis = 0)
        z1 = torch.from_numpy(z1).float().cpu()
        z2 = np.expand_dims(z2, axis = 0)
        z2 = torch.from_numpy(z2).float().cpu()

        ## selection
        selection = torch.from_numpy(selection).long().cpu()

        ##
        points_arr = self.model.G.interpolate(x = x, z1 = z1, z2 = z2, selection = selection, alpha = alpha, use_latent = use_latent).transpose(1, 2)
        points_arr = points_arr.detach().cpu().numpy()

        return points_arr



    def load_model(self):
        cat = str(self.model.opts.choice).lower()
        could_load, save_epoch = self.model.load(self.model.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

if __name__ =='__main__':
    
    point_gen = PointGenerator()
    points_arr, points_noise = point_gen.generate_pointclouds(2)
    print(points_arr.shape)
    print(points_noise.shape)