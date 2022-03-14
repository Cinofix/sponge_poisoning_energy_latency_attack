class ASICModel:
    """
    Default energy figures taken from here (45nm):
    https://ieeexplore.ieee.org/document/6757323
    Assuming all operations are 32 bit floating point (others are available).
    """

    def __init__(self, optim=True):
        # Cost of a single memory access in pJ.
        # ASSUMPTION: DRAM costs dominate (on-chip caches are negligible).
        # ASSUMPTION: reads and writes cost the same.
        self.memory_cost = 1950.0

        # Cost of a single computation (e.g. multiply) in pJ.
        # ASSUMPTION: all computations cost the same amount.
        self.compute_cost = 3.7

        # Is the hardware able to optimise memory access for sparse data?
        # ASSUMPTION: there is no overhead to (de)compression.
        self.compress_sparse_weights = optim
        self.compress_sparse_activations = optim

        # Is the hardware able to skip computations when one input is zero?
        # ASSUMPTION: zeros are uniformly distributed throughout the data.
        self.compute_skip_zero_weights = optim
        self.compute_skip_zero_activations = optim
