import ast
import json
import logging
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from tqdm import tqdm
import cv2

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def preload_dataset(cfg):

    df = pd.read_csv(cfg.DATASET.TRAIN_DATA, sep=cfg.DATASET.CSV_SEPARATOR, engine='python')
    if cfg.DATASET.CSV_IMG_KEY not in df:
        df = pd.read_csv(cfg.DATASET.TRAIN_DATA, sep='|', engine='python')
    if cfg.DATASET.CSV_IMG_KEY not in df:
        df = pd.read_csv(cfg.DATASET.TRAIN_DATA, sep=',', engine='python')
        
    image_list = df[cfg.DATASET.CSV_IMG_KEY].tolist()
    captions = df[cfg.DATASET.CSV_CAPTION_KEY].tolist()
    preload_img_data = {}
    logging.info('Preload the entire dataset...')

    # tot = ToTensor()
    
    for idx in tqdm(range(len(image_list))):
        img_dir = os.path.join(str(cfg.DATASET.IMG_DIR), str(image_list[idx]))
        
        if image_list[idx] not in preload_img_data:
            img = cv2.imread(img_dir)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            preload_img_data[image_list[idx]] = img

    print(r'The number of images is: %d '%(len(preload_img_data)))
    return preload_img_data


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_dir, img_key, caption_key, sep='\t', preload_alldata = None, is_train = True):
        logging.debug(f'Loading csv data from {input_filename}.')

        df = pd.read_csv(input_filename, sep=sep, engine='python')
        if img_key not in df:
            df = pd.read_csv(input_filename, sep='|', engine='python')
        if img_key not in df:
            df = pd.read_csv(input_filename, sep=',', engine='python')
            
        self.img_dir = img_dir
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.is_train = is_train

        self.transforms = transforms
        logging.debug('Done loading data.')

        # self.tokenize = tokenizer
        self.preload_alldata = preload_alldata

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if self.preload_alldata is not None:

            if self.images[idx] in self.preload_alldata:
                im_data = self.preload_alldata[self.images[idx]]
            else:
                im_data = Image.open(os.path.join(str(self.img_dir), str(self.images[idx])))

            images = self.transforms(im_data)
            texts = str(self.captions[idx])
        else:
            # img_name = str(self.images[idx]).split('.')[0] + '.npy'
            img_name = os.path.join(str(self.img_dir), str(self.images[idx]))
            if not os.path.exists(img_name):
                test = 1
            # img = Image.open(img_name)
            try:
                img = Image.open(os.path.join(str(self.img_dir), str(self.images[idx])))
            except:
                img = Image.open(os.path.join(str(self.img_dir), str(self.images[idx]).split('-')[0],str(self.images[idx])))
            # img = Image.fromarray(img.astype('uint8'))
            images = self.transforms(img)
            texts = str(self.captions[idx])
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, cfg, preprocess_fn, is_train):
    input_filename = cfg.DATASET.TRAIN_DATA if is_train else cfg.DATASET.VAL_DATA
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_dir=cfg.DATASET.IMG_DIR,
        img_key=cfg.DATASET.CSV_IMG_KEY,
        caption_key=cfg.DATASET.CSV_CAPTION_KEY,
        sep=cfg.DATASET.CSV_SEPARATOR,
        preload_alldata = args.preload_data if is_train else None,
        is_train = is_train,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_zeroshot_dataset(cfg, preprocess_fn, task):
    if task == 'classification':
        input_filename = cfg.DATASET.ZEROSHOT_CLS 
        imdir = cfg.DATASET.ZEROSHOT_CLS_IMDIR
        caption_key = 'label'
    elif task =='retrieval':
        input_filename = cfg.DATASET.ZEROSHOT_RET
        imdir = cfg.DATASET.ZEROSHOT_RET_IMDIR
        caption_key = 'caption'
    elif task =='po_retrieval':
        input_filename = cfg.DATASET.ZEROSHOT_PO
        imdir = cfg.DATASET.ZEROSHOT_PO_IMDIR
        caption_key = 'caption'
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_dir= imdir,
        img_key=cfg.DATASET.CSV_IMG_KEY,
        caption_key=caption_key,
        sep='\t',
        preload_alldata = None,
        is_train=False
    )
    num_samples = len(dataset)
    sampler =  None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        sampler=sampler,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args, cfg, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if cfg.DATASET.TRAIN_DATA or cfg.DATASET.TYPE == "synthetic":
        data["train"] = get_dataset_fn(cfg.DATASET.TRAIN_DATA, cfg.DATASET.TYPE)(
            args, cfg, preprocess_train, is_train=True)

    if cfg.DATASET.VAL_DATA:
        data["val"] = get_dataset_fn(cfg.DATASET.VAL_DATA, cfg.DATASET.TYPE)(
            args, cfg, preprocess_val, is_train=False)

    if cfg.DATASET.ZEROSHOT_CLS:
        data['zeroshot_cls'] = get_zeroshot_dataset(cfg, preprocess_val, 'classification')

    if cfg.DATASET.ZEROSHOT_RET:
        data['zeroshot_ret'] = get_zeroshot_dataset(cfg, preprocess_val, 'retrieval')

    if cfg.DATASET.ZEROSHOT_PO:
        data['zeroshot_po'] = get_zeroshot_dataset(cfg, preprocess_val, 'po_retrieval')

    return data

def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, cfg, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
