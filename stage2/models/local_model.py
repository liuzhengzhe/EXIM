import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from im2mesh.dvr.models import (
    decoder, depth_function
)


from .diffusers.models.attention import AttentionBlock, SpatialTransformer
from .diffusers.models.resnet import Downsample3D, FirDownsample3D, FirUpsample3D, ResnetBlock3D, Upsample3D

# 1D convolution is used for the decoder. It acts as a standard FC, but allows to use a batch of point samples features,
# additionally to the batch over the input objects.
# The dimensions are used as follows:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution is done over only all features of one point sample, this makes it a FC.


# ShapeNet Voxel Super-Resolution --------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
class ShapeNet32Vox(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNet32Vox, self).__init__()

        self.conv_1 = nn.Conv3d(1, 32, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        feature_size = (1 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim*2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)

        p2=torch.flip(p,-1)*2
        feature_0 = F.grid_sample(x, p2)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_1(x))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_1 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_2 = F.grid_sample(net, p2)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_3 = F.grid_sample(net, p2)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1024*3, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        #print (x.shape, 'x')
        x=torch.reshape(x, (x.shape[0], 1024*3))
        #print (x.shape, self.fc1.weight.shape)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=2,
                    context_dim=cross_attention_dim,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, mask=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):

            hidden_states = resnet(hidden_states, temb)


            hidden_states = attn(hidden_states, context=encoder_hidden_states, mask=mask)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class ShapeNet128Vox(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNet128Vox, self).__init__()
        # accepts 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  # out: 128

        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 64
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8



        self.conv_in_color = nn.Conv3d(1, 16, 3, padding=1)  # out: 128
        self.conv_0_color = nn.Conv3d(16, 32, 3, padding=1)  # out: 64
        self.conv_0_1_color = nn.Conv3d(32, 32, 3, padding=1)  # out: 64
        self.conv_1_color = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1_color = nn.Conv3d(64, 64, 3, padding=1)  # out: 32
        self.conv_2_color = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1_color = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3_color = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1_color = nn.Conv3d(128, 128, 3, padding=1)  # out: 8


        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        self.fc_0_color = nn.Conv1d(3+32+32+32+32 + 32 + 128  +768, hidden_dim, 1)
        self.fc_1_color = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2_color = nn.Conv1d(hidden_dim, hidden_dim, 1)

        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.fc_out_c = nn.Conv1d(hidden_dim, 3, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)


        self.conv_in_bn_color  = nn.BatchNorm3d(16)
        self.conv0_1_bn_color  = nn.BatchNorm3d(32)
        self.conv1_1_bn_color  = nn.BatchNorm3d(64)
        self.conv2_1_bn_color  = nn.BatchNorm3d(128)
        self.conv3_1_bn_color  = nn.BatchNorm3d(128)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        #print (hidden_dim, 'hidden_dim')

        self.fc0_c = nn.Conv1d(1, 32, 1)
        self.fc1_c = nn.Conv1d(16, 32, 1)
        self.fc2_c = nn.Conv1d(32, 32, 1)
        self.fc3_c = nn.Conv1d(64, 32, 1)
        self.fc4_c = nn.Conv1d(128, 32, 1)
        
        
        


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

        model_path_clip = "openai/clip-vit-large-patch14"
        #self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        self.clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float32)
        self.clip = self.clip_model.text_model.float() #.to(device=sample.device)

        #for param in self.clip.parameters():
        #  param.requires_grad=True



        '''self.clip1 = nn.Conv1d(768, 16, 1)
        self.clip2 = nn.Conv1d(768, 32, 1)
        self.clip3 = nn.Conv1d(768, 64, 1)
        self.clip4 = nn.Conv1d(768, 128, 1)
        self.clip5 = nn.Conv1d(768, 128, 1)'''
        #self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        
        
        #self.att1 = CrossAttnDownBlock3D(16, 16, 0, resnet_groups=16)
        self.att2 = CrossAttnDownBlock3D(32, 32, 0, cross_attention_dim=768) # resnet_groups=16)
        self.att3 = CrossAttnDownBlock3D(64, 64, 0, cross_attention_dim=768)# resnet_groups=16)
        self.att4 = CrossAttnDownBlock3D(128, 128, 0, cross_attention_dim=768) # resnet_groups=16)
        #self.att5 = CrossAttnDownBlock3D(128, 128, 0)
        
        
        


    def forward(self, p, x, text, pred_occ=0, pred_color=1, noise=None): 
    
        out_o, out_c=None, None


        x = x.unsqueeze(1)
        psave=p.clone()
        #print ('pred occ', pred_occ)
        if pred_occ:
          if pred_color==1:
            with torch.no_grad():
  
              p_features = p.transpose(1, -1)
              p = p.unsqueeze(1).unsqueeze(1)
  
              p2=torch.flip(p,[-1])*2
  
              p2 = torch.cat([p2 + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
              
      
              feature_0 = F.grid_sample(x, p2)  # out : (B,C (of x), 1,1,sample_num)
      
              net = self.actvn(self.conv_in(x))
              net = self.conv_in_bn(net)
              feature_1 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_0(net))
              net = self.actvn(self.conv_0_1(net))
              net = self.conv0_1_bn(net)
              feature_2 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_1(net))
              net = self.actvn(self.conv_1_1(net))
              net = self.conv1_1_bn(net)
              feature_3 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_2(net))
              net = self.actvn(self.conv_2_1(net))
              net = self.conv2_1_bn(net)
              feature_4 = F.grid_sample(net, p2)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_3(net))
              net = self.actvn(self.conv_3_1(net))
              net = self.conv3_1_bn(net)
              feature_5 = F.grid_sample(net, p2)
      
              # here every channel corresponse to one feature.
      
              features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                                   dim=1)  # (B, features, 1,7,sample_num)
              shape = features.shape
              
  
              features = torch.reshape(features,
                                       (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
              #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
      
              net = self.actvn(self.fc_0(features))
              net = self.actvn(self.fc_1(net))
              net = self.actvn(self.fc_2(net))
              #print ('net', net.shape)
              net_o = self.fc_out(net)
              out_o = net_o.squeeze(1)
    
    
          else:
  
  
              p_features = p.transpose(1, -1)
              p = p.unsqueeze(1).unsqueeze(1)
  
              p2=torch.flip(p,[-1])*2
  
              p2 = torch.cat([p2 + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
              
      
              feature_0 = F.grid_sample(x, p2)  # out : (B,C (of x), 1,1,sample_num)
      
              net = self.actvn(self.conv_in(x))
              net = self.conv_in_bn(net)
              feature_1 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_0(net))
              net = self.actvn(self.conv_0_1(net))
              net = self.conv0_1_bn(net)
              feature_2 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_1(net))
              net = self.actvn(self.conv_1_1(net))
              net = self.conv1_1_bn(net)
              feature_3 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_2(net))
              net = self.actvn(self.conv_2_1(net))
              net = self.conv2_1_bn(net)
              feature_4 = F.grid_sample(net, p2)
              net = self.maxpool(net)
      
              net = self.actvn(self.conv_3(net))
              net = self.actvn(self.conv_3_1(net))
              net = self.conv3_1_bn(net)
              feature_5 = F.grid_sample(net, p2)
      
              # here every channel corresponse to one feature.
      
              features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                                   dim=1)  # (B, features, 1,7,sample_num)
              shape = features.shape
              
  
              features = torch.reshape(features,
                                       (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
              #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
      
              net = self.actvn(self.fc_0(features))
              net = self.actvn(self.fc_1(net))
              net = self.actvn(self.fc_2(net))
              #print ('net', net.shape)
              net_o = self.fc_out(net)
              out_o = net_o.squeeze(1)
    
  
  
  

        
        if pred_color:
          #print (pred_color, 'pred color')
          mask=torch.zeros(text.shape).to('cuda')
          mask[torch.where(text!=49407)]=1
          
          #print (pred_color, 'pred colorxxxxxxxxxxxxx')
          
  
      
          encoder_hidden_states=self.clip(text).last_hidden_state
  
          clip_feature = encoder_hidden_states[:,0,:].unsqueeze(-1)  #(B,768)
  
          #x = x.unsqueeze(1)
          #print (2)
  
          p_features = psave.transpose(1, -1)

          psave = psave.unsqueeze(1).unsqueeze(1)
          p_features = p_features.unsqueeze(2).unsqueeze(2)
          #p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
          

          p2=torch.flip(psave,[-1])*2
          #print (3)
          #print (p.shape, torch.unique(p), 'ppppppppp')
          
          #print (x.shape, p2.shape, 'xp2')
          feature_0 = F.grid_sample(x, p2)  # out : (B,C (of x), 1,1,sample_num)
          
          #print (feature_0.shape, 'feature_0')
          
          feature_0 = self.fc0_c(feature_0.reshape((feature_0.shape[0], feature_0.shape[1], feature_0.shape[4]))).unsqueeze(2).unsqueeze(2)
          #print ('feature_0', feature_0.shape)
  
  
          net = self.actvn(self.conv_in_color(x))
          net = self.conv_in_bn_color(net)
          #print (4)
          
  
          
          feature_1 = F.grid_sample(net, p2)  
          feature_1 = self.fc1_c(feature_1.reshape((feature_1.shape[0], feature_1.shape[1], feature_1.shape[4]))).unsqueeze(2).unsqueeze(2)
          
          #print (feature_1.shape, 'feature_1')
          net = self.maxpool(net)
          
  
  
          net = self.actvn(self.conv_0_color(net))
          net = self.actvn(self.conv_0_1_color(net))
          net = self.conv0_1_bn_color(net)
          #print (5)
          
          
  
  
  
          feature_2 = F.grid_sample(net, p2)  # out : (B,C (of x), 1,1,sample_num)
          feature_2 = self.fc2_c(feature_2.reshape((feature_2.shape[0], feature_2.shape[1], feature_2.shape[4]))).unsqueeze(2).unsqueeze(2)
  
          net,_ = self.att2(net,encoder_hidden_states=encoder_hidden_states,mask=mask)
          #net = self.maxpool(net)
  
          net = self.actvn(self.conv_1_color(net))
          net = self.actvn(self.conv_1_1_color(net))
          net = self.conv1_1_bn_color(net)
          #print (6)

  
          feature_3 = F.grid_sample(net, p2)
          feature_3 = self.fc3_c(feature_3.reshape((feature_3.shape[0], feature_3.shape[1], feature_3.shape[4]))).unsqueeze(2).unsqueeze(2)
          
          
          net,_ = self.att3(net,encoder_hidden_states=encoder_hidden_states,mask=mask)
          #net = self.maxpool(net)
          
          net = self.actvn(self.conv_2_color(net))
          net = self.actvn(self.conv_2_1_color(net))
          net = self.conv2_1_bn_color(net)
  
          #print (pred_color, 'pred 2')
  
          feature_4 = F.grid_sample(net, p2)
          
          feature_4 = self.fc4_c(feature_4.reshape((feature_4.shape[0], feature_4.shape[1], feature_4.shape[4]))).unsqueeze(2).unsqueeze(2)
          
          
          net,_ = self.att4(net,encoder_hidden_states=encoder_hidden_states,mask=mask)
  
          net = self.actvn(self.conv_3_color(net))
          net = self.actvn(self.conv_3_1_color(net))
          net = self.conv3_1_bn_color(net)
  
  
          #feature_5 = F.grid_sample(net, p2)

          
          clip_feature=clip_feature.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,feature_4.shape[-1])
          #print (noise, 'noise')
          #if noise==None:
            

          
          #train: uncomment this. test: comment this
          noise=torch.randn(1,128,1,1,1).cuda()
          #print ('noise', noise.shape, torch.unique(noise))


          #noise=torch.randn(feature_4.shape[0], 128,1,1,1).cuda()#*5
          noise=noise.repeat(feature_4.shape[0],1,1,1,feature_4.shape[-1])
          
          #print (feature_0.shape, feature_1.shape, feature_2.shape,feature_3.shape,feature_4.shape, clip_feature.shape, 'clip')

          #print (feature_0.shape, feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, clip_feature.shape)
          #print (pred_color, 'pred 3')
          #print (torch.unique(clip_feature), torch.unique(feature_0),torch.unique(feature_2), torch.unique(feature_4), torch.unique(p_features))
          
          
          
          
          features = torch.cat((p_features, feature_0,  feature_1, feature_2, feature_3, feature_4,  noise, clip_feature),
                               dim=1)
          shape = features.shape
          

          
          features = torch.reshape(features,
                                   (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
          #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
          #print (features.shape, 'aaa')
          net = self.actvn(self.fc_0_color(features))
          net = self.actvn(self.fc_1_color(net))
          net = self.actvn(self.fc_2_color(net))
          #print ('net', net.shape)
          #net_o = self.fc_out(net)
          #out_o = net_o.squeeze(1)
  
  
          net_c = self.fc_out_c(net)
          #print ('netc1',net_c.shape, net_o.shape)
          out_c = net_c.squeeze(1)
          
          #print ('outc', out_c.shape)
          out_c=torch.permute(out_c, (0,2,1))
          #print ('outc', out_c.shape)
          
          
          out_c = torch.sigmoid(out_c)
        return out_o, out_c

    

    '''def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out'''




# ShapeNet Pointcloud Completion ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class ShapeNetPoints(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNetPoints, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):

        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out




# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

class SVR(nn.Module):


    def __init__(self, hidden_dim=256):
        super(SVR, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out