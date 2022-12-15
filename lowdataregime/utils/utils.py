import argparse
import copy
import importlib
import json
import os
import sys
import warnings
from enum import Enum
import random
from typing import Optional, Tuple, Any
import functools
import numpy as np

import torch
from filelock import FileLock


from itertools import islice

from torch.utils.data.dataloader import default_collate


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def relative_file_lock(path: str, timeout: int = -1, root_path: str = ""):
    lock_file_path = f"{os.path.join(root_path, path)}.lock"
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)
    return FileLock(lock_file_path, timeout=timeout)


def get_lock_folder(root_path):
    return os.path.join(root_path, ".lock")


def rsetattr(obj, attr, val):
    # using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        return hasattr(obj, attr)
    return functools.reduce(_hasattr, [obj] + attr.split('.'))


def remove_prefix(text, prefix):
    """
    Taken from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def str2dict(v):
    if isinstance(v, dict):
       return v

    result = {}
    for kv in v.split(","):
        k, v = kv.split("=")
        result[k] = v

    return result


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2int(v):
    if isinstance(v, str):
       if v == "None":
           return None
    return int(v)


def _get_class(class_type: str):
    """
    Get class pointer from string
    :param class_type: The type of the class as string
    :return: Class pointer
    """
    parts = class_type.split('.')
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_name)
    class_ptr = getattr(module, class_name)
    return class_ptr


def _get_class_type_as_string(obj):
    """
    Get the type of the object as string
    :param obj: The object, which class is requested
    :return: The type of the object as string
    """
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


class Serializable:
    """
    Class implements dynamic json serialization
    It can serialize any objects to json format and load back
    """

    def dumps(self) -> str:
        """
        Dumps the object to json string
        :return: The json string representation of the object
        """
        return self._to_json()

    def dumps_to_file(self, path):
        """
        Dumps the object to json file
        :param path: The path to the json file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(self.dumps())

    @classmethod
    def loads(cls, stream) -> Optional['Serializable']:
        """
        Loads the object from stream
        :param stream: The stream, which contains the json representation of the object
        :return: The loaded object
        """
        if isinstance(stream, str):
            return cls._from_json(stream)
        if hasattr(stream, "read"):
            return cls._from_json(stream.read())
        raise NotImplementedError(f"Loading from {type(stream)} is not implemented!")

    @classmethod
    def loads_from_file(cls, path):
        """
        Loads the object from file
        :param path: The path to the json file representing the object
        :return: The loaded object
        """
        return cls.loads(open(path))

    @classmethod
    def _from_dict(cls, hyperparam_dict: dict) -> Optional['Serializable']:
        """
        Creates object from dictionary
        :param hyperparam_dict: Dictionary of the members
        :return: Object with the given memebers
        """
        if hyperparam_dict is None:
            return None

        obj = cls()
        obj.__dict__.update(hyperparam_dict)
        return obj

    def _to_dict(self) -> dict:
        """
        Converts the object to dictionary
        :return: The dictionary representation of the object
        """
        return self.__dict__

    @classmethod
    def _from_json_dict(cls, frozen_dict: dict) -> Optional['Serializable']:
        """
        Loads object from json dictionary
        :param frozen_dict: The dictionary of the frozen parameters
        :return: The loaded object
        """
        if isinstance(frozen_dict, dict) and "py/obj" in frozen_dict:
            class_ptr = _get_class(frozen_dict["py/obj"])
            if class_ptr != cls:
                return class_ptr._from_json_dict(frozen_dict)
            del frozen_dict["py/obj"]
            obj = cls()

            obj.__dict__.update(
                {member_name: cls._from_json_dict(member_value)
                 for member_name, member_value in frozen_dict.items()})
        elif isinstance(frozen_dict, list):
            obj = [cls._from_json_dict(frozen_member) for frozen_member in frozen_dict]
        elif isinstance(frozen_dict, dict):
            obj = {cls._from_json_dict(frozen_member_key): cls._from_json_dict(frozen_member_value)
                   for frozen_member_key, frozen_member_value in frozen_dict.items()}
        else:
            obj = frozen_dict

        return obj

    @classmethod
    def _from_json(cls, frozen: str) -> Optional['Serializable']:
        """
        Loads object from json string
        :param frozen: The json representation of the object
        """
        if frozen is None:
            return None

        json_dict = json.loads(frozen)
        class_def = _get_class(json_dict["py/obj"])

        return class_def._from_json_dict(json_dict)

    def _value_to_json_dict(self, value):
        """
        Converts the value to json representation as dictionary
        :param value: The calue to be converted
        :return: The dictionary of the object's json representation
        """
        if isinstance(value, list):
            value = [self._value_to_json_dict(item) for item in value]
        elif isinstance(value, dict):
            value = {key: self._value_to_json_dict(item) for key, item in value.items()}
        else:
            if hasattr(value, "_to_json_dict"):
                value = value._to_json_dict()
        return value

    def _to_json_dict(self) -> dict:
        """
        Converts the object to dictionary of the json representation
        :return: The dictionary of the object's json representation
        """
        json_dict = dict()
        json_dict["py/obj"] = _get_class_type_as_string(self)

        for member_name, member_value in self._to_dict().items():
            json_dict[member_name] = self._value_to_json_dict(member_value)

        return json_dict

    def _to_json(self) -> str:
        """
        Converts the object to json string
        :return: The json string representing the object
        """
        json_dict = self._to_json_dict()

        frozen = json.dumps(json_dict, ensure_ascii=False, indent=4)
        return frozen


class SerializableEnum(Serializable, Enum):
    """Enum, which can be serialized"""

    def __init__(self, *args, **kwargs):
        # WORKAROUND: Python 3.6 does not implement the __module__ of the enum, but it is required for serialization
        if str(self.__class__.__module__) == "<unknown>":
            try:
                module = sys._getframe(2).f_globals['__name__']
                self.__class__.__module__ = module
            except (AttributeError, ValueError, KeyError) as exc:
                pass

    @classmethod
    def _from_json_dict(cls, frozen_dict: dict) -> Optional['SerializableEnum']:
        """
        Loads the enum from json dictionary
        :param frozen_dict: The dictionary of the frozen parameters
        :return: The loaded object
        """
        del frozen_dict["py/obj"]
        assert len(
            frozen_dict) == 1 and "value" in frozen_dict, f"The provided dictionary cannot be deserialized to Enum!"
        obj = cls(frozen_dict["value"])
        return obj

    def _to_json_dict(self) -> dict:
        """
        Converts the enum to dictionary of the json representation
        :return: The dictionary of the object's json representation
        """
        json_dict = dict()
        json_dict["py/obj"] = _get_class_type_as_string(self)
        json_dict["value"] = self.value

        return json_dict

    def __str__(self):
        return self.value


def pl_worker_init_function(worker_id: int, rank: Optional = None) -> None:  # pragma: no cover
    """
    The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
    set the seed with ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    Source: https://github.com/PyTorchLightning/pytorch-lightning/blob/20f37b85b68f9903df8d61e79fcebdbadacf6422/pytorch_lightning/utilities/seed.py#L91
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)

def rank_zero_only(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn

# TODO: this should be part of the cluster environment
def _get_rank() -> int:
    rank_keys = ('RANK', 'SLURM_PROCID', 'LOCAL_RANK')
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', _get_rank())


def try_plot(func):
    def wrap(self):
        try:
            func(self)
        except Exception as exp:
            self.logger.error(f"Unable to plot: {exp}!")
    return wrap


def indexed_getitem(s, index: int) -> Tuple[Any, Any]:
    return index, s.__nonindexed_getitem__(index)


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return index, self.dataset.__getitem__(index)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)


def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        ids.append(_batch[0])
        new_batch.append(_batch[1])
    return ids, default_collate(new_batch)
