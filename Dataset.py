import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from utils import sample_mask

class Cresci15(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.

        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_type = torch.load(self.root + "/edge_type.pt")
        label = torch.load(self.root + "/label.pt")
        cat_prop = torch.load(self.root + "/cat_properties_tensor.pt")
        num_prop = torch.load(self.root + "/num_properties_tensor.pt")
        des_tensor = torch.load(self.root + "/des_tensor.pt")
        tweets_tensor = torch.load(self.root + "/tweets_tensor.pt")

        features = torch.cat([cat_prop, num_prop, des_tensor, tweets_tensor], axis=1)
        data = Data(x=features, y =label, edge_index=edge_index)
        data.edge_type = edge_type


        sample_number = len(data.y)

        train_idx = torch.load(self.root + "/train_idx.pt")
        val_idx = torch.load(self.root + "/test_idx.pt")
        test_idx = torch.load(self.root + "/val_idx.pt")

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MGTAB(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.

        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_index = torch.tensor(edge_index, dtype = torch.int64)
        edge_type = torch.load(self.root + "/edge_type.pt")
        edge_weight = torch.load(self.root + "/edge_weight.pt")
        stance_label = torch.load(self.root + "/labels_stance.pt")
        bot_label = torch.load(self.root + "/labels_bot.pt")

        features = torch.load(self.root + "/features.pt")
        features = features.to(torch.float32)


        data = Data(x=features, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y1 = stance_label
        data.y2 = bot_label
        sample_number = len(data.y1)

        train_idx = range(int(0.7*sample_number))
        val_idx = range(int(0.7*sample_number), int(0.9*sample_number))
        test_idx = range(int(0.9*sample_number), int(sample_number))

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MGTABlarge(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        edge_index0 = torch.load(self.root + "/large_edge_index0.pt")
        edge_index1 = torch.load(self.root + "/large_edge_index1.pt")
        edge_index2 = torch.load(self.root + "/large_edge_index2.pt")
        edge_index3 = torch.load(self.root + "/large_edge_index3.pt")
        edge_index4 = torch.load(self.root + "/large_edge_index4.pt")
        edge_index5 = torch.load(self.root + "/large_edge_index5.pt")
        edge_index6 = torch.load(self.root + "/large_edge_index6.pt")
        edge_index = torch.cat([edge_index0, edge_index1, edge_index2, edge_index3, edge_index4, edge_index5, edge_index6], axis =1)
        
        edge_type0 = 0*torch.ones(edge_index0.shape[1], dtype=torch.int64)
        edge_type1 = 1*torch.ones(edge_index1.shape[1], dtype=torch.int64)
        edge_type2 = 2*torch.ones(edge_index2.shape[1], dtype=torch.int64)
        edge_type3 = 3*torch.ones(edge_index3.shape[1], dtype=torch.int64)
        edge_type4 = 4*torch.ones(edge_index4.shape[1], dtype=torch.int64)
        edge_type5 = 5*torch.ones(edge_index4.shape[1], dtype=torch.int64)
        edge_type6 = 6*torch.ones(edge_index4.shape[1], dtype=torch.int64)
        edge_type = torch.cat([edge_type0, edge_type1, edge_type2, edge_type3, edge_type4, edge_index5, edge_index6],axis =0)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.int64)

        stance_label = torch.load(self.root + "/labels_stance.pt")
        bot_label = torch.load(self.root + "/labels_bot.pt")
        features = torch.load(self.root + "/large_features.pt")
        x = features.to(torch.float32)

        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y1 = stance_label
        data.y2 = bot_label

        labeled_sample_number = len(data.y1)
        all_sample_number = data.x.shape[0]

        train_idx = range(int(0.7*labeled_sample_number))
        val_idx = range(int(0.7*labeled_sample_number), int(0.9*labeled_sample_number))
        test_idx = range(int(0.9*labeled_sample_number), int(labeled_sample_number))

        data.train_mask = sample_mask(train_idx, all_sample_number)
        data.val_mask = sample_mask(val_idx, all_sample_number)
        data.test_mask = sample_mask(test_idx, all_sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MGTABNew(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'sample_uids_10k.csv',
            'node_meta_10k.json',
            'num_properties_tensor_10k.pt',
            'tweets_tensor_10k.pt',
            'profile_image_feats_10k.pt',
            'image_feats_10k.pt',
            'edge_index_10k.pt',
            'edge_type_10k.pt',
            'edge_weight_10k.pt',
            'label_10k.pt',
            'train_mask_10k.pt',
            'valid_mask_10k.pt',
            'test_mask_10k.pt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        rd = self.root
        # Node features
        prop    = torch.load(rd + '/num_properties_tensor_10k.pt')
        tweet   = torch.load(rd + '/tweets_tensor_10k.pt')
        profile = torch.load(rd + '/profile_image_feats_10k.pt')
        images  = torch.load(rd + '/image_feats_10k.pt')

        x = torch.cat([prop, tweet], dim=1)

        # Graph
        edge_index  = torch.load(rd + '/edge_index_10k.pt')
        edge_type   = torch.load(rd + '/edge_type_10k.pt')
        edge_weight = torch.load(rd + '/edge_weight_10k.pt')

        # Labels and masks
        label      = torch.load(rd + '/label_10k.pt')
        train_mask = torch.load(rd + '/train_mask_10k.pt')
        val_mask   = torch.load(rd + '/valid_mask_10k.pt')
        test_mask  = torch.load(rd + '/test_mask_10k.pt')

        # assemble Data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type   = edge_type
        data.edge_weight = edge_weight
        data.y  = label
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        data.img        =  torch.cat([profile, images], dim=1)


        # filters and transform
        data_list = [data]
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MGTABAblateImages(InMemoryDataset):
    # run from RGT_ff.py
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'sample_uids_10k.csv',
            'node_meta_10k.json',
            'num_properties_tensor_10k.pt',
            'tweets_tensor_10k.pt',
            'profile_image_feats_10k.pt',
            'image_feats_10k.pt',
            'edge_index_10k.pt',
            'edge_type_10k.pt',
            'edge_weight_10k.pt',
            'label_10k.pt',
            'train_mask_10k.pt',
            'valid_mask_10k.pt',
            'test_mask_10k.pt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        rd = self.root
        # Node features
        prop    = torch.load(rd + '/num_properties_tensor_10k.pt')
        tweet   = torch.load(rd + '/tweets_tensor_10k.pt')
        profile = torch.load(rd + '/profile_image_feats_10k.pt')
        # images  = torch.load(rd + '/image_feats_10k.pt')

        x = torch.cat([prop, tweet], dim=1)

        # Graph
        edge_index  = torch.load(rd + '/edge_index_10k.pt')
        edge_type   = torch.load(rd + '/edge_type_10k.pt')
        edge_weight = torch.load(rd + '/edge_weight_10k.pt')

        # Labels and masks
        label      = torch.load(rd + '/label_10k.pt')
        train_mask = torch.load(rd + '/train_mask_10k.pt')
        val_mask   = torch.load(rd + '/valid_mask_10k.pt')
        test_mask  = torch.load(rd + '/test_mask_10k.pt')

        # assemble Data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type   = edge_type
        data.edge_weight = edge_weight
        data.y  = label
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        data.img        =  profile


        # filters and transform
        data_list = [data]
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class MGTABAblateProfile(InMemoryDataset):
    # run from RGT_ff.py
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'sample_uids_10k.csv',
            'node_meta_10k.json',
            'num_properties_tensor_10k.pt',
            'tweets_tensor_10k.pt',
            'profile_image_feats_10k.pt',
            'image_feats_10k.pt',
            'edge_index_10k.pt',
            'edge_type_10k.pt',
            'edge_weight_10k.pt',
            'label_10k.pt',
            'train_mask_10k.pt',
            'valid_mask_10k.pt',
            'test_mask_10k.pt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        rd = self.root
        # Node features
        prop    = torch.load(rd + '/num_properties_tensor_10k.pt')
        tweet   = torch.load(rd + '/tweets_tensor_10k.pt')
        # profile = torch.load(rd + '/profile_image_feats_10k.pt')
        # images  = torch.load(rd + '/image_feats_10k.pt')

        x = torch.cat([prop, tweet], dim=1)

        # Graph
        edge_index  = torch.load(rd + '/edge_index_10k.pt')
        edge_type   = torch.load(rd + '/edge_type_10k.pt')
        edge_weight = torch.load(rd + '/edge_weight_10k.pt')

        # Labels and masks
        label      = torch.load(rd + '/label_10k.pt')
        train_mask = torch.load(rd + '/train_mask_10k.pt')
        val_mask   = torch.load(rd + '/valid_mask_10k.pt')
        test_mask  = torch.load(rd + '/test_mask_10k.pt')

        # assemble Data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type   = edge_type
        data.edge_weight = edge_weight
        data.y  = label
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        data.img        =  images


        # filters and transform
        data_list = [data]
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MGTABAblateProfileandImages(InMemoryDataset):
    # run from RGT.py
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'sample_uids_10k.csv',
            'node_meta_10k.json',
            'num_properties_tensor_10k.pt',
            'tweets_tensor_10k.pt',
            'profile_image_feats_10k.pt',
            'image_feats_10k.pt',
            'edge_index_10k.pt',
            'edge_type_10k.pt',
            'edge_weight_10k.pt',
            'label_10k.pt',
            'train_mask_10k.pt',
            'valid_mask_10k.pt',
            'test_mask_10k.pt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        rd = self.root
        # Node features
        prop    = torch.load(rd + '/num_properties_tensor_10k.pt')
        tweet   = torch.load(rd + '/tweets_tensor_10k.pt')
        # profile = torch.load(rd + '/profile_image_feats_10k.pt')
        # images  = torch.load(rd + '/image_feats_10k.pt')

        x = torch.cat([prop, tweet], dim=1)

        # Graph
        edge_index  = torch.load(rd + '/edge_index_10k.pt')
        edge_type   = torch.load(rd + '/edge_type_10k.pt')
        edge_weight = torch.load(rd + '/edge_weight_10k.pt')

        # Labels and masks
        label      = torch.load(rd + '/label_10k.pt')
        train_mask = torch.load(rd + '/train_mask_10k.pt')
        val_mask   = torch.load(rd + '/valid_mask_10k.pt')
        test_mask  = torch.load(rd + '/test_mask_10k.pt')

        # assemble Data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type   = edge_type
        data.edge_weight = edge_weight
        data.y  = label
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        # data.img        =  torch.cat([profile, images], dim=1)


        # filters and transform
        data_list = [data]
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])