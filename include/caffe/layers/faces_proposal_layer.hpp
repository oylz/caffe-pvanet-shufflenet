#ifndef CAFFE_XYZ_FACES_PROPOSAL_LAYER_HPP_
#define CAFFE_XYZ_FACES_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class FacesProposalLayer : public Layer<Dtype> {
public:
    explicit FacesProposalLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
	 _conf_thresh = 0;
	 _nms_thresh = 0;
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){}
    virtual inline const char* type() const { return "FacesProposal"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

private:
    void ForwardEmpty(const vector<Blob<Dtype>*>& top);
 
private:
	float _conf_thresh;
	float _nms_thresh;
};

}  // namespace caffe



#endif


