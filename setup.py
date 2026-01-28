from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure we can find the source files
src_dir = os.path.join(os.path.dirname(__file__), 'src')

setup(
    name='optimulti_fusion',
    ext_modules=[
        CUDAExtension('optimulti_fusion_cuda', [
            os.path.join(src_dir, 'fusion_kernel.cu'),
            os.path.join(src_dir, 'bindings.cpp'),
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
