#include <algorithm>
#include <vector>

#include "caffe/layers/faces_proposal_layer.hpp"

namespace caffe {
/*
template <typename Dtype>
__global__ void FacesProposalDoForward(const int nthreads,
    const Dtype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}
*/


template <typename Dtype>
void FacesProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#if 1
	Forward_cpu(bottom, top);
#else
	int face_count = 2;
	top[0]->Reshape(face_count, 5, 1, 1); // faces:face_countx5
	Dtype* top0 = top[0]->mutable_cpu_data();
	float tmp[10] = {
			0, 10, 20, 30, 40,
			0, 50, 60, 70, 80
		};
	caffe_copy(10, (Dtype*)tmp, top0);


	Dtype* top1 = top[1]->mutable_cpu_data();
	float fc = face_count;
	caffe_copy(1, (Dtype*)&fc, top1);
    	/*FacesProposalDoForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        	count, bias_data, num, channels,
        	top_height, top_width, top_data);
	*/
#endif
}


INSTANTIATE_LAYER_GPU_FUNCS(FacesProposalLayer);



}
