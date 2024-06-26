import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess/23654936#23654936

    example usage to debug a torch distributed run on rank 0:
    if torch.distributed.get_rank() == 0:
        from fairchem.core.common.debug import ForkedPdb
        ForkedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin