import torch
import torch.nn.functional as F
import numpy as np
import copy
import pywt
from spconv.core import AlgoHint, ConvAlgo
import spconv.pytorch as spconv
from spconv.pytorch.hash import HashTable
from models.module.dwt_utils import prep_filt_sfb3d, prep_filt_afb3d
from models.module.diffusion_network import UNetModel, MyUNetModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(input_size, kernel_size=1, stride=1, pad=0):
    from math import floor, ceil
    h = floor( ((input_size + (2 * pad) - kernel_size )/ stride) + 1)
    return h

def indices_to_key(keys, spatial_size, delta = 50): # HACK
    new_keys = keys[:, 3] + keys[:, 2] * (spatial_size[-1]+delta) + keys[:, 1] * (spatial_size[-1]+delta) * (spatial_size[-2]+delta) + \
    keys[:, 0] * (spatial_size[-1]+delta) * (spatial_size[-2]+delta) * (spatial_size[-3]+delta)

    return new_keys
def create_coordinates(resolution, feature_dim = 1):
    dimensions_samples = np.linspace(0, resolution - 1, resolution)

    if feature_dim > 1:
        feature_samples = np.arange(feature_dim)
        d, x, y, z = np.meshgrid(feature_samples, dimensions_samples, dimensions_samples, dimensions_samples)
        d, x, y, z = np.swapaxes(d[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(x[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(y[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(z[:, :, :, :, np.newaxis], 0, 1)
        coordinates = np.concatenate((d, x, y, z), axis=4)
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates
    else:
        x, y, z = np.meshgrid(dimensions_samples, dimensions_samples, dimensions_samples)
        x, y, z = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]
        coordinates = np.concatenate((x, y, z), axis=3)
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()
    def forward(self, x):
        return x

### DISCRIMINATOR
class Discriminator(torch.nn.Module):
    def __init__(self, i_dim, d_dim, z_dim):
        super(Discriminator, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.i_dim = i_dim

        self.conv_1 = torch.nn.Conv3d(self.i_dim,             self.d_dim,    4, stride=1, padding=0, bias=True)
        self.conv_2 = torch.nn.Conv3d(self.d_dim,    self.d_dim*2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = torch.nn.Conv3d(self.d_dim*2,  self.d_dim*4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = torch.nn.Conv3d(self.d_dim*4,  self.d_dim*8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = torch.nn.Conv3d(self.d_dim*8,  self.d_dim*16, 3, stride=1, padding=0, bias=True)
        self.conv_6 = torch.nn.Conv3d(self.d_dim*16, self.z_dim,    1, stride=1, padding=0, bias=True)

    def forward(self, voxels):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = torch.sigmoid(out)

        return out

class MultiScaleMLP(torch.nn.Module):
    def __init__(self, config, data_num, J, shape_list = None):
        super().__init__()
        self.J = J
        self.data_num = data_num
        self.config = config


        ## initialize layers
        if hasattr(self.config, 'use_diffusion') and self.config.use_diffusion:
            self.low_layer = UNetModel(in_channels=1,
                                model_channels=self.config.unet_model_channels,
                                out_channels=2 if hasattr(self.config, 'diffusion_learn_sigma') and self.config.diffusion_learn_sigma else 1,
                                num_res_blocks=self.config.unet_num_res_blocks,
                                channel_mult=self.config.unet_channel_mult_low,
                                attention_resolutions=self.config.attention_resolutions,
                                dropout=0,
                                dims=3,
                                activation = self.config.unet_activation if hasattr(self.config, 'unet_activation') else None)
        elif hasattr(self.config, 'use_conv3d') and self.config.use_conv3d:
            self.low_layer = Conv3D(self.config.latent_dim // (self.J + 1), 1, config)
        else:
            self.low_layer = MLP(3 + self.config.latent_dim // (self.J + 1), 1, config)


        if hasattr(self.config, 'use_diffusion') and self.config.use_diffusion:
            self.highs_layers = torch.nn.ModuleList([UNetModel(in_channels=2,
                                model_channels=self.config.unet_model_channels,
                                out_channels= 2 if hasattr(self.config, 'diffusion_learn_sigma') and self.config.diffusion_learn_sigma else 1,
                                num_res_blocks=self.config.unet_num_res_blocks,
                                channel_mult=self.config.unet_channel_mult,
                                attention_resolutions=self.config.attention_resolutions,
                                dropout=0,
                                dims=3) for i in range(self.J)])
        elif hasattr(self.config, 'highs_use_conv3d') and self.config.highs_use_conv3d:
            input_latent_dims = self.config.latent_dim // (self.J + 1)
            if hasattr(self.config, 'highs_use_unent') and self.config.highs_use_unent:
                if hasattr(self.config, 'highs_no_code_unet') and self.config.highs_no_code_unet:
                    input_latent_dims = 0
                self.highs_layers = torch.nn.ModuleList(
                    [MyUNetModel(in_channels= input_latent_dims + 1,
                                spatial_size= shape_list[i][0],
                                model_channels=self.config.unet_model_channels,
                                out_channels= 1,
                                num_res_blocks=self.config.unet_num_res_blocks,
                                channel_mult=self.config.unet_channel_mult,
                                attention_resolutions=self.config.attention_resolutions,
                                dropout=0,
                                dims=3) for i in range(self.J)])
            else:
                if hasattr(self.config, 'highs_use_downsample_features') and self.config.highs_use_downsample_features:
                    input_latent_dims += self.config.downsample_features_dim
                    assert shape_list is not None
                    self.low_to_high_features_conv = DownSampleConv3D(1, self.config.downsample_features_dim, shape_list[-1][0], config)

                self.highs_layers = torch.nn.ModuleList(
                    [ Conv3DHigh(input_latent_dims, 1, i, config) for i in range(self.J)])
        else:
            self.highs_layers = torch.nn.ModuleList([MLP(3 + self.config.latent_dim // (self.J + 1), 1, config) for i in range(self.J)])

        ## intialize codes
        self.latent_codes = torch.nn.Embedding(num_embeddings=data_num, embedding_dim=self.config.latent_dim) # old
        self.latent_codes.weight = torch.nn.Parameter(self.latent_codes.weight * self.config.code_bound, requires_grad = True) ## for setting bound
        #self.latent_codes = torch.nn.Parameter(torch.normal(0, config.code_bound, (data_num, self.config.latent_dim)), requires_grad = True)

        if hasattr(self.config, 'use_VAD') and self.config.use_VAD:
            self.latent_codes_log_var = torch.nn.Embedding(num_embeddings=data_num, embedding_dim=self.config.latent_dim)  # old
            self.latent_codes_log_var.weight = torch.nn.Parameter(self.latent_codes.weight * self.config.code_bound,
                                                          requires_grad=True)  ## for setting bound

        ## saved_low
        self.saved_low_pred = None
        self.saved_highs_pred = [None] * self.J

        ## zero tensors
        self.zero_tensors = torch.zeros((1, 1), requires_grad = False).float().to(device)
        self.zero_tensors_grid = torch.zeros((1, 1, 1), requires_grad = False).float().to(device)
        self.shape_list = shape_list

        ## remove unsued things
        if hasattr(self.config, 'remove_reductant') and self.config.remove_reductant:
            for i in range(self.J + 1):
                if (hasattr(self.config, 'zero_stages')  and i in self.config.zero_stages) or \
                        (hasattr(self.config, 'gt_stages') and i in self.config.gt_stages):
                    if i == self.J:
                        self.low_layer = None
                    else:
                        self.highs_layers[i] = None

    def extract_full_coeff(self, code_indices, level, stage, zero_stages = [],
                gt_stages = [], gt_low = None, gt_highs = None, VAD_codes = None):

        if hasattr(self.config, 'use_VAD') and self.config.use_VAD:
            if VAD_codes is not None:
                codes = VAD_codes
            else:
                codes_mean = self.latent_codes(code_indices)
                codes_log_var = self.latent_codes_log_var(code_indices)
                codes_std = torch.exp(0.5 * codes_log_var)
                eps = torch.randn_like(codes_std)
                codes = codes_mean + eps * codes_std
        else:
            if code_indices is None:
                codes = None
            else:
                codes = self.latent_codes(code_indices)
        split_dim = self.config.latent_dim // (self.J + 1)
        assert self.shape_list is not None
        spatial_shape = self.shape_list[level]

        assert self.shape_list is not None
        if level in zero_stages:
            new_zeros = self.zero_tensors.unsqueeze(0).unsqueeze(0).expand(code_indices.size(0), -1, self.shape_list[level][0], self.shape_list[level][1], self.shape_list[level][2])
            return new_zeros

        if level in gt_stages:
            assert gt_low is not None and gt_highs is not None
            if level == self.J:
                return gt_low.detach()
            else:
                return gt_highs[level].detach()

        ## other case
        if hasattr(self.config, 'use_conv3d') and self.config.use_conv3d and level == self.J:
            if hasattr(self.config, 'new_low_fix') and self.config.new_low_fix:
                codes = codes[:, level * split_dim:(level + 1) * split_dim]
            else:
                codes = codes[:, :split_dim]
            output_results = self.low_layer(codes, spatial_shape)
        elif  level < self.J and hasattr(self.config, 'highs_use_conv3d') and self.config.highs_use_conv3d:

            if codes is not None:
                codes = codes[:, level * split_dim:(level + 1) * split_dim]

            ## if use conditional
            if hasattr(self.config, 'highs_use_unent') and self.config.highs_use_unent:
                upsampled_low = F.interpolate(gt_low, size = tuple(spatial_shape))
                if hasattr(self.config, 'highs_no_code_unet') and self.config.highs_no_code_unet:
                    inputs = upsampled_low
                else:
                    codes = codes.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, spatial_shape[0], spatial_shape[1], spatial_shape[2])
                    inputs = torch.cat((upsampled_low, codes), dim = 1)
                output_results = self.highs_layers[level](inputs)
            elif hasattr(self.config, 'highs_use_downsample_features') and self.config.highs_use_downsample_features:
                low_codes = self.low_to_high_features_conv(gt_low)
                codes = torch.cat((codes, low_codes), dim=1)

                output_results = self.highs_layers[level](codes, spatial_shape)
            else:
                output_results = self.highs_layers[level](codes, spatial_shape)
        else:
            raise Exception("MLPs not support full predictions due to memory problems.....")

        if self.config.train_only_current_level:
            if level != stage:
                output_results = output_results.detach()  # remove gradients
        else:
            if level < stage:
                output_results = output_results.detach() # remove gradients

        return output_results

    def forward(self, indices, level, code_indices, stage, spatial_shape, zero_stages = [],
                gt_stages = [], save_low = False, save_high = False, gt_low = None, gt_highs = None):
        if hasattr(self.config, 'use_VAD') and self.config.use_VAD:
            codes_mean = self.latent_codes(code_indices)
            codes_log_var = self.latent_codes_log_var(code_indices)
            codes_std = torch.exp(0.5 * codes_log_var)
            eps = torch.randn_like(codes_std)
            codes = codes_mean + eps * codes_std
        else:
            codes = self.latent_codes(code_indices)
        split_dim = self.config.latent_dim // (self.J + 1)
        indices = indices.long()

        ## return zeros coefficient
        if level in zero_stages:
            return self.zero_tensors.expand(indices.size(0), -1)

        ## return the gt coefficient
        if level in gt_stages:
            assert gt_low is not None and gt_highs is not None
            if level == self.J:
                return gt_low.detach()[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
            else:
                return gt_highs[level].detach()[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]

        if hasattr(self.config, 'use_conv3d') and self.config.use_conv3d and level == self.J:

            if self.saved_low_pred is not None:
                dense_results = torch.from_numpy(self.saved_low_pred).float().to(device)
            else:
                if hasattr(self.config, 'new_low_fix') and self.config.new_low_fix:
                    codes = codes[:, level*split_dim:(level+1)*split_dim]
                else:
                    codes = codes[:, :split_dim]
                dense_results = self.low_layer(codes, spatial_shape)

            output_results = dense_results[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
            if save_low:
                self.saved_low_pred = dense_results.detach().cpu().numpy()
        else:

            if level < self.J and hasattr(self.config, 'highs_use_conv3d') and self.config.highs_use_conv3d:
                if self.saved_highs_pred[level] is not None:
                    dense_results = torch.from_numpy(self.saved_highs_pred[level]).float().to(device)
                else:
                    codes = codes[:, level*split_dim:(level+1)*split_dim]

                    ## if use conditional
                    if hasattr(self.config, 'highs_use_unent') and self.config.highs_use_unent:
                        upsampled_low = F.interpolate(gt_low, size=tuple(spatial_shape))
                        if hasattr(self.config, 'highs_no_code_unet') and self.config.highs_no_code_unet:
                            inputs = upsampled_low
                        else:
                            codes = codes.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, spatial_shape[0],
                                                                                        spatial_shape[1], spatial_shape[2])
                            inputs = torch.cat((upsampled_low, codes), dim=1)
                        dense_results = self.highs_layers[level](inputs)
                    elif hasattr(self.config, 'highs_use_downsample_features') and self.config.highs_use_downsample_features:
                        low_codes = self.low_to_high_features_conv(gt_low)
                        codes = torch.cat((codes, low_codes), dim = 1)

                        dense_results = self.highs_layers[level](codes, spatial_shape)
                    else:
                        dense_results = self.highs_layers[level](codes, spatial_shape)
                output_results = dense_results[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
                if save_high:
                    self.saved_highs_pred[level] = dense_results.detach().cpu().numpy()
            else:
                ## slice
                slice_codes = codes[indices[:, 0], level*split_dim:(level+1)*split_dim]
                coordinates = indices[:, 1:]
                if self.config.scale_coordinates:
                    spatial_shape = torch.from_numpy(np.array(spatial_shape)).float().unsqueeze(0).to(device)
                    coordinates = coordinates / spatial_shape
                input_features = torch.cat((slice_codes, coordinates), dim = 1)


                if level == self.J:
                    output_results = self.low_layer(input_features)
                else:
                    output_results = self.highs_layers[level](input_features)

        if self.config.train_only_current_level:
            if level != stage:
                output_results = output_results.detach()  # remove gradients
        else:
            if level < stage:
                output_results = output_results.detach() # remove gradients

        return output_results

class NearestUpsample3D(torch.nn.Module):
    def __init__(self, upsample_ratio):
        super().__init__()
        self.upsample_ratio = upsample_ratio

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upsample_ratio, mode='nearest')
        return x

class DownSampleConv3D(torch.nn.Module):
    def __init__(self, input_dim, output_dim, spatial_size, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim

        feature_size = spatial_size
        for layer_dim, kernel_size, stride in self.config.conv3d_downsample_tuple_layers:
            layer_list = []

            if stride[0] == 1:
                conv_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                kernel_size=kernel_size, padding='same')
            else:
                conv_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                kernel_size=kernel_size, stride=stride)
            layer_list.append(conv_layer)

            if stride[0] != 1:
                feature_size = conv_output_shape(feature_size, kernel_size[0], stride[0], pad=0)

            if self.config.use_instance_norm:
                norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                layer_list.append(norm_layer)
            if self.config.use_layer_norm:
                norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size],
                                                elementwise_affine=self.config.use_layer_affine)
                layer_list.append(norm_layer)



            new_layer = torch.nn.Sequential(
                *layer_list
            )

            self.layers.append(new_layer)
            current_dim = layer_dim

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)


    def forward(self, input_features):

        x = input_features
        batch_size = x.size(0)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.last_layer(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view((batch_size, -1))

        return x

class Conv3DHigh(torch.nn.Module):
    def __init__(self, input_dim, output_dim, level, config):
        super().__init__()
        self.config = config
        self.desne_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim
        for layer_dim in self.config.conv3d_dense_layers:
            new_layer = torch.nn.Linear(current_dim, layer_dim)
            self.desne_layers.append(new_layer)
            current_dim = layer_dim
        self.desne_layers.append(torch.nn.Linear(current_dim, self.config.conv3d_latent_dim))


        ### conv3d layers
        current_dim = self.config.conv3d_latent_dim // 8

        feature_size = 2
        conv3d_tuple_layers = copy.deepcopy(self.config.conv3d_tuple_layers)
        for i in range(self.config.max_depth - level):
            conv3d_tuple_layers.extend(copy.deepcopy(self.config.conv3d_tuple_layers_highs_append))

        for layer_dim, kernel_size, stride in conv3d_tuple_layers:
            if self.config.conv3d_use_upsample:

                if stride[0] > 1:
                    layer_list = [NearestUpsample3D(stride)]
                else:
                    layer_list = []
                layer_list.append(torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                    kernel_size=kernel_size, padding = 'same'))
                feature_size = int(feature_size * stride[0])
                if self.config.use_instance_norm:
                    norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                    layer_list.append(norm_layer)
                if self.config.use_layer_norm:
                    norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size], elementwise_affine=self.config.use_layer_affine)
                    layer_list.append(norm_layer)

                new_layer = torch.nn.Sequential(
                    *layer_list
                )

            else:
                new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim,
                                                     kernel_size=kernel_size,
                                                     stride=stride)
            self.layers.append(new_layer)
            current_dim = layer_dim

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        ### layer initialization
        for layer in self.desne_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
            torch.nn.init.constant_(layer.bias, 0)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)



    def forward(self, codes, spatial_shape):

        ## transform and reshape
        batch_size = codes.size(0)
        x = codes
        for layer in self.desne_layers:
            x = layer(x)
            x = self.activation(x)

        ## re shape
        x = x.view(batch_size, -1, 2, 2, 2)

        ## upsamples
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        ##  last layer
        x = self.last_layer(x)

        low_bound = x.size(2) // 2 - spatial_shape[0] // 2, x.size(3) // 2 - spatial_shape[1] // 2, x.size(4) // 2 - spatial_shape[2] // 2
        delta = spatial_shape[0] % 2, spatial_shape[1] % 2, spatial_shape[2] % 2
        high_bound = x.size(2) // 2 + spatial_shape[0] // 2 + delta[0], x.size(3) // 2 + spatial_shape[1] // 2 + delta[1], x.size(4) // 2 + spatial_shape[2] // 2 + delta[2]
        x = x[:, :, low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]]

        return x

class Conv3D(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.config = config
        self.desne_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim
        for layer_dim in self.config.conv3d_dense_layers:
            new_layer = torch.nn.Linear(current_dim, layer_dim)
            self.desne_layers.append(new_layer)
            current_dim = layer_dim
        self.desne_layers.append(torch.nn.Linear(current_dim, self.config.conv3d_latent_dim))


        ### conv3d layers
        current_dim = self.config.conv3d_latent_dim // 8

        feature_size = 2
        if hasattr(self.config, 'conv3d_tuple_layers'):
            for layer_dim, kernel_size, stride in self.config.conv3d_tuple_layers:
                if hasattr(self.config, 'conv3d_use_upsample') and self.config.conv3d_use_upsample:

                    layer_list = [ NearestUpsample3D(stride),
                        torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                        kernel_size=kernel_size, padding = 'same')]
                    feature_size *= stride[0]
                    if self.config.use_instance_norm:
                        norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                        layer_list.append(norm_layer)
                    if self.config.use_layer_norm:
                        norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size], elementwise_affine=self.config.use_layer_affine)
                        layer_list.append(norm_layer)

                    new_layer = torch.nn.Sequential(
                        *layer_list
                    )

                else:
                    new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim,
                                                         kernel_size=kernel_size,
                                                         stride=stride)
                self.layers.append(new_layer)
                current_dim = layer_dim
        else:
            for layer_dim in self.config.conv3d_layers:
                new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim, kernel_size=self.config.conv3d_kernel_size,
                                                         stride = (2, 2, 2))
                self.layers.append(new_layer)
                current_dim = layer_dim

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        ### layer initialization
        for layer in self.desne_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
            torch.nn.init.constant_(layer.bias, 0)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)



    def forward(self, codes, spatial_shape):

        ## transform and reshape
        batch_size = codes.size(0)
        x = codes
        for layer in self.desne_layers:
            x = layer(x)
            x = self.activation(x)

        ## re shape
        x = x.view(batch_size, -1, 2, 2, 2)

        ## upsamples
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        ##  last layer
        x = self.last_layer(x)

        low_bound = x.size(2) // 2 - spatial_shape[0] // 2, x.size(3) // 2 - spatial_shape[1] // 2, x.size(4) // 2 - spatial_shape[2] // 2
        delta = spatial_shape[0] % 2, spatial_shape[1] % 2, spatial_shape[2] % 2
        high_bound = x.size(2) // 2 + spatial_shape[0] // 2 + delta[0], x.size(3) // 2 + spatial_shape[1] // 2 + delta[1], x.size(4) // 2 + spatial_shape[2] // 2 + delta[2]
        x = x[:, :, low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]]

        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()


        ###
        if self.config.use_fourier_features:
            current_dim = self.config.linear_layers[0] * 2 + input_dim - 3
        else:
            current_dim = input_dim

        self.first_layer = torch.nn.Linear(current_dim, self.config.linear_layers[0])
        current_dim = self.config.linear_layers[0]

        for layer_dim in self.config.linear_layers[1:]:
            linear_layer = torch.nn.Linear(current_dim, layer_dim)
            self.layers.append(linear_layer)
            current_dim = layer_dim

        self.last_layer = torch.nn.Linear(current_dim, output_dim)
        self.activation = self.config.activation

        ### layer initialization
        torch.nn.init.normal_(self.first_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.first_layer.bias, 0)

        for layer in self.layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
            torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)

        if self.config.use_fourier_features:
            self.B = torch.nn.Parameter(torch.randn((3, self.config.linear_layers[0])) * self.config.fourier_norm, requires_grad = False)





    def forward(self, x):

        if self.config.use_fourier_features:
            x_cos = torch.cos(torch.matmul(x[...,-3:], self.B) * 2 * torch.pi)
            x_sin = torch.sin(torch.matmul(x[...,-3:], self.B) * 2 * torch.pi)
            x = torch.cat((x[...,:-3], x_cos, x_sin), dim = -1)

        x = self.first_layer(x)
        x = self.activation(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.last_layer(x)

def get_conv_shape(current_spatial_shape, conv_module):
    spatial_shape_out = spconv.ops.get_conv_output_size(current_spatial_shape, kernel_size=conv_module.kernel_size, stride = conv_module.stride, padding=conv_module.padding,
                                    dilation=conv_module.dilation)

    return spatial_shape_out

def get_conv_indices(current_indices, current_spatial_shape, conv_module, batch_size):
    indices_out = spconv.ops.get_indice_pairs(indices=current_indices, batch_size = batch_size, spatial_shape=current_spatial_shape,
                                algo= ConvAlgo.Native, ksize=conv_module.kernel_size, stride=conv_module.stride, padding = conv_module.padding,
                                          dilation=conv_module.dilation, out_padding = conv_module.output_padding)[0]
    spatial_shape_out = spconv.ops.get_conv_output_size(current_spatial_shape, kernel_size=conv_module.kernel_size, stride = conv_module.stride, padding=conv_module.padding,
                                    dilation=conv_module.dilation)
    return indices_out, spatial_shape_out

def compute_modules(conv_dim, input_shape, h0, g0, mode):


    assert mode in ['zero', 'constant']

    N = input_shape[conv_dim]
    L = h0.numel()
    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # padding for input
    input_shape = copy.deepcopy(input_shape)
    if p % 2 == 1:
        input_shape[conv_dim] += 1

    kernel_size = [1, 1, 1]
    kernel_size[conv_dim] = L
    stride = [1, 1, 1]
    stride[conv_dim] = 2
    pad = [0, 0, 0]
    pad[conv_dim] = p // 2
    conv_module = spconv.SparseConv3d(in_channels = 1, out_channels = 1, kernel_size = kernel_size,
                                      stride= stride, padding=pad, bias = False, groups = 1).to(device)
    conv_module.weight = torch.nn.Parameter(torch.reshape(h0 , conv_module.weight.size()).to(device), requires_grad = False)
    pad = [0, 0, 0]
    pad[conv_dim] = L - 2
    inv_module = spconv.SparseConvTranspose3d(in_channels = 1, out_channels = 1, kernel_size = kernel_size,
                                      stride= stride, padding=pad, bias = False, groups = 1).to(device)
    #g0 = torch.flip(g0, dims = [2+conv_dim])
    inv_module.weight = torch.nn.Parameter(torch.reshape(g0 , inv_module.weight.size()).to(device), requires_grad = False)

    output_shape = get_conv_shape(input_shape, conv_module)

    return output_shape, conv_module, inv_module

def initalize_modules(input_shape, max_depth, h0_dep, h0_col, h0_row, g0_dep, g0_col, g0_row, mode):


    ## compute input_indices
    shapes_list = [input_shape]
    current_shape = input_shape
    conv_modules, inv_modules = [], []

    assert mode in ["zero", 'constant']

    ## compute shapes and indices
    for i in range(max_depth):
        current_shape, conv_module_row, inv_module_row = compute_modules(conv_dim = 2, input_shape = current_shape, h0=h0_row, g0 = g0_row, mode = mode)
        current_shape, conv_module_col, inv_module_col = compute_modules(conv_dim = 1, input_shape = current_shape, h0=h0_col, g0 = g0_col, mode = mode)
        current_shape, conv_module_dep, inv_module_dep = compute_modules(conv_dim = 0, input_shape = current_shape, h0=h0_dep, g0 = g0_dep, mode = mode)
        shapes_list.append(current_shape)
        conv_modules.append([conv_module_row, conv_module_col, conv_module_dep])
        inv_modules.append([inv_module_dep, inv_module_col, inv_module_row])

    return shapes_list, conv_modules, inv_modules

class SparseComposer(torch.nn.Module):
    def __init__(self, input_shape, J=1, wave='db1', mode='zero', inverse_dwt_module = None):
        super().__init__()
        self.inverse_dwt_module = inverse_dwt_module
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode
        self.input_shape = input_shape

        ## Need for inverse
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])

        ### initalize module
        self.shape_list, self.conv_modules, self.inv_modules = initalize_modules(input_shape = input_shape, max_depth = self.J,
                          h0_dep = self.h0_dep, h0_col = self.h0_col, h0_row = self.h0_row,
                          g0_dep = self.g0_dep, g0_col = self.g0_col, g0_row = self.g0_row,
                          mode = self.mode)


    def forward(self, input_indices, weight_func, **kwargs):

        batch_size, indices_list = self.extract_indcies_list(input_indices)

        ### compute the features from bottom-up
        if self.inverse_dwt_module is None:
            current_coeff = None
            for i in range(self.J)[::-1]:
                kwargs['spatial_shape'] = self.shape_list[i+1]
                output_coeff = weight_func(indices = indices_list[i+1], level = i+1, **kwargs)

                ### add with previous layer
                if current_coeff is not None:
                    current_coeff = current_coeff.unsqueeze(1) + output_coeff
                else:
                    current_coeff = output_coeff

                current_coeff = spconv.SparseConvTensor(features = current_coeff, indices = indices_list[i+1],
                                        spatial_shape = self.shape_list[i+1], batch_size=batch_size)



                ### perform idwf
                current_coeff = self.inv_modules[i][0](current_coeff)
                current_coeff = self.inv_modules[i][1](current_coeff)
                current_coeff = self.inv_modules[i][2](current_coeff)

                ### retrived only useful coeff
                table = HashTable(device, torch.int32, torch.float32, max_size=current_coeff.indices.size(0) * 2)
                coeff_indices, query_indices = indices_to_key(current_coeff.indices, spatial_size=current_coeff.spatial_shape),\
                                               indices_to_key(indices_list[i], spatial_size=current_coeff.spatial_shape)


                table.insert(coeff_indices, current_coeff.features)
                current_coeff, isempty = table.query(query_indices)

                assert sum(isempty) == 0

            kwargs['spatial_shape'] = self.shape_list[0]
            output_coeff = weight_func(indices=indices_list[0], level= 0, **kwargs)
            final_coeff = current_coeff.unsqueeze(1) + output_coeff
        else:
            final_coeff = None
            low, highs = None, []
            for i in range(self.J)[::-1]:
                kwargs['spatial_shape'] = self.shape_list[i+1]
                output_coeff = weight_func(indices = indices_list[i+1], level = i+1, **kwargs)
                current_coeff = spconv.SparseConvTensor(features = output_coeff, indices = indices_list[i+1],
                                        spatial_shape = self.shape_list[i+1], batch_size=batch_size)
                dense_coeff = current_coeff.dense(channels_first = True)
                if i+1 == self.J:
                    low = dense_coeff
                else:
                    highs = [dense_coeff] + highs

            ## last layers
            kwargs['spatial_shape'] = self.shape_list[0]
            output_coeff = weight_func(indices=indices_list[0], level=0, **kwargs)
            current_coeff = spconv.SparseConvTensor(features=output_coeff, indices=indices_list[0],
                                                    spatial_shape=self.shape_list[0], batch_size=batch_size)
            dense_coeff = current_coeff.dense(channels_first=True)
            highs = [dense_coeff] + highs

            final_coeff = self.inverse_dwt_module((low, highs))
            indices_long = indices_list[0].long()
            final_coeff = final_coeff[indices_long[:, 0], 0, indices_long[:, 1], indices_long[:, 2], indices_long[:, 3]].unsqueeze(1)

        return final_coeff

    def extract_indcies_list(self, input_indices):
        ## prepare the indices
        batch_size = input_indices.size(0)
        sample_num = input_indices.size(1)
        batch_indices = torch.arange(0, batch_size).int()
        batch_indices = batch_indices.unsqueeze(1).repeat((1, sample_num)).view((-1, 1)).to(device)
        input_indices = input_indices.view((-1, 3))
        current_indices = torch.cat((batch_indices, input_indices), dim=-1)
        ## compute the indices for each level
        indices_list = [current_indices]
        current_shape = self.input_shape
        for i in range(self.J):
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][0],
                                                              batch_size=batch_size)
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][1],
                                                              batch_size=batch_size)
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][2],
                                                              batch_size=batch_size)
            indices_list.append(current_indices)
        return batch_size, indices_list


if __name__ == "__main__":

    from configs import config
    from models.module.dwt import DWTForward3d, DWTInverse3d


    module = spconv.SparseConvTranspose3d(1, 1, (16, 1, 1), stride = (2, 1, 1), groups= 1, indice_key= 'spconv3').to(device)
    module_2 = spconv.SparseConvTranspose3d(1, 1, (1, 16, 1), stride = (1, 2, 1), groups= 1).to(device)
    module_3 = spconv.SparseConvTranspose3d(1, 1, (1, 1, 16), stride = (1, 1, 2), groups= 1).to(device)
    module.weight = torch.nn.Parameter(torch.zeros_like(module.weight), requires_grad = False)
    #print(module.weight)
    #print(module.weight.size())

    resolution = 64

    features =  torch.zeros(resolution * resolution * resolution, 1).to(device)# your features with shape [N, num_channels]
    indices =  create_coordinates(resolution,  1).view(-1, 3).int().to(device)# your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
    indices = torch.cat((torch.zeros((indices.size(0), 1), dtype = torch.int32).to(device),indices), dim = -1)
    spatial_shape =  [resolution, resolution, resolution]# spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
    batch_size = 1 # batch size of your sparse tensor.
    x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

    #print(x)
    #print(indices)
    #print(x_dense_NCHW.size())

    x_out = module(x)
    inverse_module = spconv.SparseInverseConv3d(1, 1, (16, 1, 1), indice_key = 'spconv3').to(device)
    x_out_inverse = inverse_module(x_out)
    #print(x_out)
    #print(x_out.indices.size())

    table = HashTable(device, torch.int32, torch.float, max_size=x_out.indices.size(0) * 2)
    table.insert(x_out.indices,x_out.features)

    vq, _ = table.query(x_out.indices)
    #print(vq)
    #print(x_out.spatial_shape)

    indices_conv = spconv.ops.get_indice_pairs(indices=x.indices, batch_size = 1, spatial_shape=x.spatial_shape,
                                algo= ConvAlgo.Native, ksize=module.kernel_size, stride=module.stride, padding = module.padding,
                                          dilation=module.dilation, out_padding = module.output_padding)
    #print(indices_conv)
    x_output_size = spconv.ops.get_conv_output_size(x.spatial_shape, kernel_size=module.kernel_size, stride = module.stride, padding=module.padding,
                                    dilation=module.dilation)
    #print(x_output_size)





    #dwt_forward_3d = DWTForward3d(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    #dwt_inverse_3d = DWTInverse3d(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    #network = MultiScaleMLP(config = config, data_num = 1, dwt_module = dwt_forward_3d, inverse_dwt_module = dwt_inverse_3d).to(device)

    #indices = torch.from_numpy(np.arange(1)).to(device)
    #output = network(indices)
