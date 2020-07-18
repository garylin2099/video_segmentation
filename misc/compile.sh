# compile the bilinear warping operator

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# nvcc -std=c++11 -c -o bilinear_warping.cu.o bilinear_warping.cu.cc \
# 	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

# # g++ -std=c++11 -shared -o bilinear_warping.so bilinear_warping.cc \
# # 	bilinear_warping.cu.o -I $TF_INC -fPIC -L  /usr/local/cuda/lib64/ -lcudart
# g++ -std=c++11 -shared -o bilinear_warping.so bilinear_warping.cc \
# 	bilinear_warping.cu.o -I $TF_INC -fPIC -L  /usr/local/cuda-10.0/lib64/ -lcudart
# # g++ -std=c++11 -shared -o bilinear_warping.so bilinear_warping.cc \
# # 	bilinear_warping.cu.o -I $TF_INC -fPIC -L  /usr/local/cuda-10.0/lib64/ -lcudart -D_GLIBCXX_USE_CXX11_ABI=0


TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o bilinear_warping.cu.o bilinear_warping.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o bilinear_warping.so bilinear_warping.cc \
  bilinear_warping.cu.o ${TF_CFLAGS[@]} -fPIC -L /usr/local/cuda-10.0/lib64 -lcudart ${TF_LFLAGS[@]}