import os
import logging
from torch.utils import data
import numpy as np
import yaml
import random
logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None,
                 shared_dict={}, n_views=24, cache_fields=False,
                 split_model_for_images=False):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            shared_dict (dict): shared dictionary (used for field caching)
            n_views (int): number of views (only relevant when using field
                caching)
            cache_fields(bool): whether to cache fields; this option can be
                useful for small overfitting experiments
            split_model_for_images (bool): whether to split a model by its
                views (can be relevant for small overfitting experiments to
                       perform validation on all views)
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cache_fields = cache_fields
        self.n_views = n_views
        self.cached_fields = shared_dict
        self.split_model_for_images = split_model_for_images

        if split_model_for_images:
            assert(n_views > 0)
            print('You are splitting the models by images. Make sure that you entered the correct number of views.')

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]
        categories.sort()
        categories=['03001627']
        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx
        self.dic=np.load('../data/blip_balance500.npy',allow_pickle=1)[()]
        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, str(split) + '.lst')

            if not os.path.exists(split_file):
                models_c = [f for f in os.listdir(
                    subpath) if os.path.isdir(os.path.join(subpath, f))]
            else:
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            
            
            '''models_c=[]
            for m in models_c_ori:
                #print (m)
                if str(m[0])=='b' or str(m[0])=='a' or str(m[0])=='9' or str(m[0])=='8' or str(m[0])=='7':
                  models_c.append(m)'''
                  
            if split_model_for_images:
                for m in models_c:
                    #print ('m',m,flush=True)
                    if os.path.exists('../data/03001627_train/'+m+'.npy')==0 or m not in self.dic.keys(): # or os.path.exists('../ifnet/shapenet/data/03001627/'+m+'/boundary_{}_samples.npz')==0:
                      continue
                    #if m!='1006be65e7bc937e9141f9b58470d646':
                    #  continue
                    for i in range(n_views):
                        self.models += [
                            {'category': c, 'model': m,
                                'category_id': c_idx, 'image_id': i}
                        ]
            else:
              for m in models_c:
                  #print ('m',m,flush=True)
                  if os.path.exists('../data/03001627_train/'+m+'.npy')==0 or m not in self.dic.keys(): #or os.path.exists('../ifnet/shapenet/data/03001627/'+m+'/boundary_0.1_samples.npz')==0 or m not in self.dic.keys():
                    continue

                  self.models += [
                      {'category': c, 'model': m, 'category_id': c_idx}
                  ]
        #self.models=[{'category': '03001627', 'model': 'bd6a8b133fa4d269491d6cee03fef2a9', 'category_id': 0},{'category': '03001627', 'model': 'bd6a5c01b9c6f17a82db9fca4b68095', 'category_id': 0},{'category': '03001627', 'model': 'bd0b06e158bcee8ac0d89fc15154c9a2', 'category_id': 0},{'category': '03001627', 'model': 'bbdaf1503b9e2219df6cfab91d65bb91', 'category_id': 0},{'category': '03001627', 'model': 'bbba083270a2b0d031d7d27dc50ba701', 'category_id': 0},{'category': '03001627', 'model': 'b9027939e6c71d844d256d962a5df83b', 'category_id': 0}]
        #self.models=[{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0},{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0},{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0},{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0},{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0},{'category': '03001627', 'model': '1033ee86cc8bac4390962e4fb7072b86', 'category_id': 0}]
        self.num_sample_points = 50000
        num_workers = 10

        sample_distribution = [0.5,0.5]
        sample_sigmas = [0.01,0.1]
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)
        self.num_samples = np.rint(self.sample_distribution * self.num_sample_points).astype(np.uint32)
        
        #print(self.num_samples, self.sample_distribution , num_sample_points)




 

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}
        for field_name, field in self.fields.items():
            try:
                if self.cache_fields:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                    else:
                        idx_img = np.random.randint(0, self.n_views)
                    k = '%s_%s_%d' % (model_path, field_name, idx_img)

                    if k in self.cached_fields:
                        field_data = self.cached_fields[k]
                    else:
                        field_data = field.load(model_path, idx, c_idx,
                                                input_idx_img=idx_img)
                        self.cached_fields[k] = field_data
                else:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                        field_data = field.load(
                            model_path, idx, c_idx, idx_img)
                    else:
                        field_data = field.load(model_path, idx, c_idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occurred when loading field %s of model %s (%s)'
                        % (field_name, model, category)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)






        
        input = np.load('../data/03001627_train/'+model+'.npy')
        '''points = []
        coords = []
        occupancies = []
        for i, num in enumerate(self.num_samples):
            boundary_samples_path = '../ifnet/shapenet/data/03001627/'+model+'/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
            boundary_samples_npz = np.load(boundary_samples_path,allow_pickle=True)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])
        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points


        data['grid_coords']=np.array(coords, dtype=np.float32)
        data['occupancies']=np.array(occupancies, dtype=np.float32)
        data['points']=np.array(points, dtype=np.float32)'''
        data['inputs']=np.array(input, dtype=np.float32)
        
        
        
        text_idx=random.randint(0,len(self.dic[model])-1)
        text=self.dic[model][text_idx]
        
        text=' '.join(text.split(' ')[:20])
        data['text']=text
        
        
        
        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
