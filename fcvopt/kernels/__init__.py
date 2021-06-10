from gpytorch.kernels import RBFKernel
from .matern import Matern32Kernel,Matern52Kernel
from .hamming_kernel import HammingKernel, ZeroOneKernel
from .constant_kernel import ConstantKernel
from .multitaskkernel import MultiTaskKernel