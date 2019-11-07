# Since we could ensure that multi-proces would write into the different parts
# For efficiency, we use Multiprocess.RawArray

from torch.multiprocessing import RawArray
from multiprocessing.managers import BaseProxy
import numpy as np

from torchrl.replay_buffers.base import BaseReplayBuffer

from .shmarray import NpShmemArray

from .shmarray import get_random_tag

class SharedBaseReplayBuffer(BaseReplayBuffer):
    """
    Basic Replay Buffer
    """
    def __init__(self, 
        max_replay_buffer_size,
        worker_nums
        # example_dict,
        # tag
    ):
        super().__init__(max_replay_buffer_size)

        self.worker_nums = worker_nums
        if not hasattr(self, "tag"):
            self.tag = get_random_tag()
            
        # raw_size = RawArray('i', self.worker_nums)
        # # self._size = [ 0 for _ in range(self.worker_nums)]
        # self._size = np.frombuffer(raw_size, dtype=np.int32)
        # raw_top = RawArray('i', self.worker_nums)
        # # self._top = [ 0 for _ in range(self.worker_nums)]
        # self._top = np.frombuffer(raw_top, dtype=np.int32)

    def build_by_example(self, example_dict):
        self._size  = NpShmemArray(self.worker_nums, np.int32, self.tag+"_size")
        self._top   = NpShmemArray(self.worker_nums, np.int32, self.tag+"_top")

        self.tags = {}
        self.shapes = {}
        for key in example_dict:
            if not hasattr( self, "_" + key ):
                current_tag = "_"+key
                self.tags[current_tag] = self.tag+current_tag
                shape = (self._max_replay_buffer_size, self.worker_nums) + \
                    np.shape(example_dict[key])
                self.shapes[current_tag] = shape
                
                # mat_size = shape[0] * shape[1] * shape[2] 
                # buffer = RawArray('d', mat_size )
                np_array = NpShmemArray(shape, np.float32, self.tag+current_tag)
                self.__setattr__(current_tag, np_array )

    def rebuild_from_tag(self):

        self._size  = NpShmemArray(self.worker_nums, np.int32, 
            self.tag+"_size", create=False)
        self._top   = NpShmemArray(self.worker_nums, np.int32,
            self.tag+"_top", create=False)

        for key in self.tags:
            # print(key, self.tags[key])
            np_array = NpShmemArray(self.shapes[key], np.float32,
                self.tags[key], create=False)
            self.__setattr__(key, np_array )

    def add_sample(self, sample_dict, worker_rank, **kwargs):
        for key in sample_dict:
            # print(key, sample_dict[key])
            self.__getattribute__( "_" + key )[self._top[worker_rank], worker_rank] = sample_dict[key]
            # print(self.__getattribute__( "_" + key )[self._top[worker_rank], worker_rank])
        # print(self._size)
        self._advance(worker_rank)

    def terminate_episode(self):
        pass

    def _advance(self, worker_rank):
        self._top[worker_rank] = (self._top[worker_rank] + 1) % \
            self._max_replay_buffer_size
        # print(self._size[worker_rank], self._size[1 - worker_rank], self._max_replay_buffer_size)
        # print(self._size)
        if self._size[worker_rank] < self._max_replay_buffer_size:
            self._size[worker_rank] = self._size[worker_rank] + 1

    def random_batch(self, batch_size, sample_key):
        size = self.num_steps_can_sample()
        # print(size)
        indices = np.random.randint(0, size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[indices].reshape(
                ( batch_size * self.worker_nums,-1))
        return return_dict

    def num_steps_can_sample(self):
        # print(self._obs.shape)
        # print(self._obs[1])
        min_size = np.min(self._size)
        max_size = np.max(self._size)
        assert max_size == min_size, \
            "all worker should gather the same amount of samples"
        return min_size


# class SharedBaseReplayBuffer(BaseReplayBuffer):
#     """
#     Basic Replay Buffer
#     """
#     def __init__(self, 
#         max_replay_buffer_size,
#         worker_nums,
#         example_dict
#     ):
#         super().__init__(max_replay_buffer_size)
#         self.worker_nums = worker_nums
#         raw_size = RawArray('i', self.worker_nums)
#         # self._size = [ 0 for _ in range(self.worker_nums)]
#         self._size = np.frombuffer(raw_size, dtype=np.int32)
#         raw_top = RawArray('i', self.worker_nums)
#         # self._top = [ 0 for _ in range(self.worker_nums)]
#         self._top = np.frombuffer(raw_top, dtype=np.int32)

#         for key in example_dict:
#             if not hasattr( self, "_" + key ):
#                 shape = (self._max_replay_buffer_size, self.worker_nums) + \
#                     np.shape(example_dict[key])
#                 mat_size = shape[0] * shape[1] * shape[2] 
#                 buffer = RawArray('d', mat_size )
#                 self.__setattr__( "_" + key,
#                     np.frombuffer(buffer).reshape(shape) )

#     def add_sample(self, sample_dict, worker_rank, **kwargs):
#         for key in sample_dict:
#             # if not hasattr( self, "_" + key ):
#             #     shape = (self._max_replay_buffer_size, self.worker_nums) + \
#             #         np.shape(sample_dict[key])
#             #     mat_size = shape[0] * shape[1] * shape[2] 
#             #     buffer = RawArray('d', mat_size )
#             #     self.__setattr__( "_" + key,
#             #         np.frombuffer(buffer).reshape(shape) )
#             self.__getattribute__( "_" + key )[self._top[worker_rank], worker_rank] = sample_dict[key].copy() 
#         self._advance(worker_rank)

#     def terminate_episode(self):
#         pass

#     def _advance(self, worker_rank):
#         self._top[worker_rank] = (self._top[worker_rank] + 1) % \
#             self._max_replay_buffer_size
#         # print(self._size[worker_rank], self._size[1 - worker_rank], self._max_replay_buffer_size)
#         # print(self._size)
#         if self._size[worker_rank] < self._max_replay_buffer_size:
#             self._size[worker_rank] = self._size[worker_rank] + 1

#     def random_batch(self, batch_size, sample_key):
#         size = self.num_steps_can_sample()
#         # print(size)
#         indices = np.random.randint(0, size, batch_size)
#         return_dict = {}
#         for key in sample_key:
#             return_dict[key] = self.__getattribute__("_"+key)[indices].reshape(
#                 ( batch_size * self.worker_nums,-1))
#         return return_dict

#     def num_steps_can_sample(self):
#         # print(self._obs.shape)
#         # print(self._obs[1])
#         min_size = np.min(self._size)
#         max_size = np.max(self._size)
#         assert max_size == min_size, \
#             "all worker should gather the same amount of samples"
#         return min_size
