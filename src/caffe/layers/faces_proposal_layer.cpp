#include "caffe/layers/faces_proposal_layer.hpp"
#include <Eigen>
//#include<StdVector>
#include <string>
#include <string>

/*this(RowMajor) will cause build error
typedef Eigen::Matrix<float, 100, 2, Eigen::RowMajor> FH2;
typedef Eigen::Matrix<float, 100, 8, Eigen::RowMajor> FH8;
typedef Eigen::Matrix<float, 100, 4, Eigen::RowMajor> FH4;
typedef Eigen::Matrix<float, 100, 5, Eigen::RowMajor> FH5;
typedef Eigen::Matrix<float, 1, 6, Eigen::RowMajor> FO6;
typedef Eigen::Matrix<float, 100, 1, Eigen::RowMajor> FH1;
typedef Eigen::Matrix<float, -1, 5, Eigen::RowMajor> MH5;
*/
typedef Eigen::Matrix<float, 100, 2> FH2;
typedef Eigen::Matrix<float, 100, 8> FH8;
typedef Eigen::Matrix<float, 100, 4> FH4;
typedef Eigen::Matrix<float, 100, 5> FH5;
typedef Eigen::Matrix<float, 1, 6> FO6;
typedef Eigen::Matrix<float, 100, 1> FH1;
typedef Eigen::Matrix<float, -1, 5> MH5;
typedef Eigen::Matrix<float, 1, 5> FH5LINE;

#define F2E(in, out)\
	for(int i = 0; i < (out).rows(); i++){\
		for(int j = 0; j < (out).cols(); j++){\
			(out)(i, j) = (float)*((in)+pos);\
			pos++;\
		}\
	}	

#define mmax(a, b) (a)>(b)?(a):(b)
#define mmin(a, b) (a)<(b)?(a):(b)
//#define GENLOG

struct fnms_dataN{
	FH5LINE data;
	int index;
};
typedef boost::shared_ptr<fnms_dataN> fnms_data;


float fiou(const FH5LINE &d1, const FH5LINE &d2, const std::string &type) {
  	float x1 = mmax(d1(0), d2(0));
  	float y1 = mmax(d1(1), d2(1));
  	float x2 = mmin(d1(2), d2(2));
  	float y2 = mmin(d1(3), d2(3));

  	if (x1 <= x2 && y1 <= y2) {
    		float i = (x2 - x1 + 1) * (y2 - y1 + 1);
    		float u;
    		if (type == ""){
      			u = (d1(2) - d1(0) + 1) * (d1(3) - d1(1) + 1) +
          			(d2(2) - d2(0) + 1) * (d2(3) - d2(1) + 1) - i;
		}
    		else if (type == "min"){
      			u = mmin((d1(2) - d1(0) + 1) * (d1(3) - d1(1) + 1),
              			(d2(2) - d2(0) + 1) * (d2(3) - d2(1) + 1));
		}
    		else{
      			throw std::runtime_error(std::string("unsupported IOU type ") + type);
		}
    		return i / u;
  	} 
   	return 0;
}
inline bool fcmp(const fnms_data &d1, const fnms_data &d2){
	return d1->data(4) < d2->data(4);
}

std::vector<int> fnms(const FH5 &det,
                            const float &threshold, const std::string &type = "") {
    	std::vector<fnms_data> dets;
    	for (int i = 0; i < (int)det.rows(); i++) {
      		fnms_data item(new fnms_dataN());
      		item->data = det.row(i);
      		item->index = i;
      		dets.push_back(item);
    	}
	// confidence is stored at col idx 4
    	std::sort(dets.begin(), 
		dets.end(), fcmp);

    	std::vector<int> ids;
    	int pick_count = 0;
    	int total = dets.size();
    	std::vector<bool> picked(total, false);
    	int pick_idx = total - 1;

    	while (pick_count < total) {
      		while (picked[pick_idx]){
        		pick_idx--;
		}
      		ids.push_back(dets[pick_idx]->index);
      		picked[pick_idx] = true;
      		pick_count++;
      		for (int i = 0; i < pick_idx; i++) {
        		if (picked[i] == false &&
            			fiou(dets[i]->data, dets[pick_idx]->data, type) > threshold) {
          			picked[i] = true;
          			pick_count++;
        		}
      		}
    	}
    	std::vector<int> res;
    	for (int i = 0; i < (int)ids.size(); i++){
      		res.push_back(ids[i]);
	}
    	return res;
}








FH8  bbox_transform_inv(const FH4 &boxes, const FH8 &deltas){
    //if boxes.shape[0] == 0:
     //   return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

	FH1 widths;
	for(int i = 0; i < 100; i++){
		widths(i) = boxes(i, 2) - boxes(i, 0) + 1;
	}	
	FH1 heights(100, 1);
	for(int i = 0; i < 100; i++){
		heights(i) = boxes(i, 3) - boxes(i, 1) + 1;
	}

	FH1 ctr_x(100, 1);
	for(int i = 0; i < 100; i++){
		ctr_x(i) = boxes(i, 0) + 0.5*widths(i);
	}
	FH1 ctr_y(100, 1);
	for(int i = 0; i < 100; i++){
		ctr_y(i) = boxes(i, 1) + 0.5*heights(i);
	}

	FH2 dx(100, 2);
	FH2 dy(100, 2);
	FH2 dw(100, 2);
	FH2 dh(100, 2);
	for(int i = 0; i < 100; i++){
		dx(i, 0) = deltas(i, 0); dx(i, 1) = deltas(i, 0+4);
		dy(i, 0) = deltas(i, 1); dy(i, 1) = deltas(i, 1+4);
		dw(i, 0) = deltas(i, 2); dw(i, 1) = deltas(i, 2+4);
		dh(i, 0) = deltas(i, 3); dh(i, 1) = deltas(i, 3+4);
	}

	FH2 pred_ctr_x(100, 2);
	FH2 pred_ctr_y(100, 2);
	FH2 pred_w(100, 2);
	FH2 pred_h(100, 2);
	for(int i = 0; i < 100; i++){
		pred_ctr_x(i, 0) = dx(i, 0)*widths(i) + ctr_x(i);
		pred_ctr_x(i, 1) = dx(i, 1)*widths(i) + ctr_x(i);
		
		pred_ctr_y(i, 0) = dy(i, 0)*heights(i) + ctr_y(i);
		pred_ctr_y(i, 1) = dy(i, 1)*heights(i) + ctr_y(i);
		
		pred_w(i, 0) = exp(dw(i, 0)) * widths(i);
		pred_w(i, 1) = exp(dw(i, 1)) * widths(i);
		
		pred_h(i, 0) = exp(dh(i, 0)) * heights(i);
		pred_h(i, 1) = exp(dh(i, 1)) * heights(i);
	}

	FH8 pred_boxes(100, 8);
	for(int i = 0; i < 100; i++){
		int pos = 0;
		pred_boxes(i, pos) = pred_ctr_x(i, 0) - 0.5*pred_w(i, 0);
		pred_boxes(i, pos+4) = pred_ctr_x(i, 1) - 0.5*pred_w(i, 1);

		
		pos = 1;
		pred_boxes(i, pos) = pred_ctr_y(i, 0) - 0.5*pred_h(i, 0);
		pred_boxes(i, pos+4) = pred_ctr_y(i, 1) - 0.5*pred_h(i, 1);


		pos = 2;
		pred_boxes(i, pos) = pred_ctr_x(i, 0) + 0.5*pred_w(i, 0);
		pred_boxes(i, pos+4) = pred_ctr_x(i, 1) + 0.5*pred_w(i, 1);


		pos = 3;
		pred_boxes(i, pos) = pred_ctr_y(i, 0) + 0.5*pred_h(i, 0);
		pred_boxes(i, pos+4) = pred_ctr_y(i, 1) + 0.5*pred_h(i, 1);
	}

	return pred_boxes;	
}

FH8 clip_boxes(const FH8 &boxes, float im_shape0, float im_shape1){

	FH8 re(100, 8);
	for(int i = 0; i < 100; i++){
		// 0,4
		int pos = 0;	
		float tmp1 = mmax(mmin(boxes(i, pos), im_shape1), 0);		
		float tmp2 = mmax(mmin(boxes(i, pos+4), im_shape1), 0);		
		re(i, pos) = tmp1;
		re(i, pos+4) = tmp2;

		// 1,5
		pos = 1;
		tmp1 = mmax(mmin(boxes(i, pos), im_shape0), 0);		
		tmp2 = mmax(mmin(boxes(i, pos+4), im_shape0), 0);		
		re(i, pos) = tmp1;
		re(i, pos+4) = tmp2;

		// 2,6
		pos = 2;	
		tmp1 = mmax(mmin(boxes(i, pos), im_shape1), 0);		
		tmp2 = mmax(mmin(boxes(i, pos+4), im_shape1), 0);		
		re(i, pos) = tmp1;
		re(i, pos+4) = tmp2;

		//3,7
		pos = 3;
		tmp1 = mmax(mmin(boxes(i, pos), im_shape0), 0);		
		tmp2 = mmax(mmin(boxes(i, pos+4), im_shape0), 0);		
		re(i, pos) = tmp1;
		re(i, pos+4) = tmp2;
	}
	return re;
}


namespace caffe{

template <typename Dtype>
void FacesProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
	_conf_thresh = this->layer_param_.faces_proposal_param().conf();
	_nms_thresh = this->layer_param_.faces_proposal_param().nms();
	top[0]->Reshape(1, 5, 1, 1);
	top[1]->Reshape(1, 1, 1, 1);
#ifdef GENLOG
	LOG(INFO) << "FacesProposalLayer LayerSetup(" << _conf_thresh <<
		" ," << _nms_thresh << ")";
#endif
}

template <typename Dtype>
void FacesProposalLayer<Dtype>::ForwardEmpty(const vector<Blob<Dtype>*>& top){
	int faces_count = 0;
	MH5 faces(1, 5);
	faces(0) = 0;	
	faces(1) = 0;
	faces(2) = 0;
	faces(3) = 0;
	
       	// reshape, forward            
	Dtype* top0 = top[0]->mutable_cpu_data();

	caffe_copy(1, (Dtype*)faces.data(), top0);

	Dtype* top1 = top[1]->mutable_cpu_data();
	float fc = faces_count;
	caffe_copy(1, (Dtype*)&fc, top1);
}

template <typename Dtype>
void FacesProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
//#ifdef GENLOG
	LOG(INFO) << "FacesProposalLayer begin";
//#endif

#if 1
	const Dtype *cls_prob = bottom[0]->cpu_data(); // 100x2
	const Dtype *bbox_pred = bottom[1]->cpu_data(); // 100x8
	const Dtype *rois = bottom[2]->cpu_data(); // 100x5
	const Dtype *im_info = bottom[3]->cpu_data();//[0];todo // 6

	int pos = 0;
	FH2 fcls_prob(100, 2);
	F2E(cls_prob, fcls_prob)

	pos = 0;
	FH8 fbbox_pred(100, 8);
	F2E(bbox_pred, fbbox_pred)

	pos = 0;
	FH5 frois(100, 5);
	F2E(rois, frois)

	pos = 0;
	FO6 fim_info(1, 6);
	F2E(im_info, fim_info)
	
	const float EPSINON = 0.000000001;
	for(int i = 0; i < fim_info.size(); i++){
		float tmp = fim_info(i);
		if(-EPSINON<tmp && tmp<EPSINON){
			// empty	
			return ForwardEmpty(top);
		}
	}
	// unscale back to raw image space
	FH4 boxes(100, 4);
	for(int i = 0; i < 100; i++){
		for(int j = 0; j < 4; j++){
			boxes(i, j) = frois(i, j+1)/fim_info(j+2);
		}
	}
#ifdef GENLOG
	LOG(INFO) << "boxes:H" << boxes << "H";
#endif

        // Apply bounding-box regression deltas
	FH8 pred_boxes = bbox_transform_inv(boxes, fbbox_pred);	
#ifdef GENLOG
	LOG(INFO) << "pred_boxes:H\n" << pred_boxes << "H";
#endif

	float im_shape0 = fim_info(0)/fim_info(2);
	float im_shape1 = fim_info(1)/fim_info(3);
	pred_boxes = clip_boxes(pred_boxes, im_shape0, im_shape1);
#ifdef GENLOG
	LOG(INFO) << "pred_boxes1:H\n" << pred_boxes << "H";
#endif

        // nms, thresh                
	//FH4 cls_boxes;
	//FH1 cls_scores;
	FH5 dets0(100, 5);
	for(int i = 0; i < 100; i++){
		/*int pos = 0;
		cls_boxes(i, pos) = pred_boxes(i, pos+4);
		pos = 1;
		cls_boxes(i, pos) = pred_boxes(i, pos+4);
		pos = 2;
		cls_boxes(i, pos) = pred_boxes(i, pos+4);
		pos = 3;
		cls_boxes(i, pos) = pred_boxes(i, pos+4);
		//
		cls_scores(i) = fcls_prob(i, 1);
		*/
		int pos = 0;
		dets0(i, pos) = pred_boxes(i, pos+4);
		pos = 1;
		dets0(i, pos) = pred_boxes(i, pos+4);
		pos = 2;
		dets0(i, pos) = pred_boxes(i, pos+4);
		pos = 3;
		dets0(i, pos) = pred_boxes(i, pos+4);
		//
		dets0(i, 4) =  fcls_prob(i, 1);

	} 
#ifdef GENLOG
	LOG(INFO) << "dets:H\n" << dets0 << "H";
#endif
		
	std::vector<int> keep = fnms(dets0, _nms_thresh);
	MH5 dets(keep.size(), 5);
#ifdef GENLOG
	LOG(INFO) << "keep:H\n";
#endif
	for(int i = 0; i < keep.size(); i++){
		int row = keep[i];
#ifdef GENLOG
		LOG(INFO) << "\t-------keep:" << keep[i];
#endif
		dets.row(i) = dets0.row(row);
	}
#ifdef GENLOG
	LOG(INFO) << "H";
	LOG(INFO) << "newdets:H\n" << dets << "H";
#endif
	std::vector<int> inds;
#ifdef GENLOG
		LOG(INFO) << "inds:H";
#endif

	for(int row = 0; row < dets.rows(); row++){
		float tmp = dets(row, 4);
		if(tmp > _conf_thresh){
#ifdef GENLOG
			LOG(INFO) << "\t------inds:" << row;
#endif
			inds.push_back(row);
		}
	} 	
#ifdef GENLOG
		LOG(INFO) << "H";
#endif
	// 
	int faces_count = inds.size();
	if(faces_count == 0){
		return ForwardEmpty(top);
	}

#ifdef GENLOG
	{
	MH5 faces;
		if(faces_count > 0){	
        		faces = MH5(faces_count, 5);
			for(int i = 0; i < faces_count; i++){
				faces(i, 0) = 0;
				int row = inds[i];
				faces(i, 1) = dets(row, 0) * fim_info(2);
				faces(i, 2) = dets(row, 1) * fim_info(3);
				faces(i, 3) = dets(row, 2) * fim_info(4);
				faces(i, 4) = dets(row, 3) * fim_info(5);
			}	
		}
		LOG(INFO) << "faces:H\n" << faces << "H";
	}	
#endif

	float *buf = new float[faces_count*5];
	pos = 0;
	for(int i = 0; i < faces_count; i++){
		int row = inds[i];

		*(buf + pos) = 0;
		pos++;

		*(buf + pos) = dets(row, 0) * fim_info(2);	
		pos++;

		*(buf + pos) = dets(row, 1) * fim_info(3);	
		pos++;

		*(buf + pos) = dets(row, 2) * fim_info(4);	
		pos++;

		*(buf + pos) = dets(row, 3) * fim_info(5);	
		pos++;
	}
        // reshape, forward            
	top[0]->Reshape(faces_count, 5, 1, 1); // faces:faces_countx5
	Dtype* top0 = top[0]->mutable_cpu_data();

	caffe_copy(faces_count*5, (Dtype*)buf, top0);
	

	Dtype* top1 = top[1]->mutable_cpu_data();
	float fc = faces_count;
	caffe_copy(1, (Dtype*)&fc, top1);

	delete []buf;
#else
	int faces_count = 2;
	//Dtype* top0 = top[0]->mutable_cpu_data();
	top[0]->Reshape(faces_count, 5, 1, 1); // faces:faces_countx5
	Dtype* top0 = top[0]->mutable_cpu_data();

	float tmp[10] = {
			0, 10, 20, 30, 40,
			0, 50, 60, 70, 80
		};
	caffe_copy<Dtype>(10, (Dtype*)tmp, top0);

	Dtype* top1 = top[1]->mutable_cpu_data();
	float fc = faces_count;
	caffe_copy(1, (Dtype*)&fc, top1);

#endif
}


#ifdef CPU_ONLY
STUB_GPU(FacesProposalLayer);
#endif

INSTANTIATE_CLASS(FacesProposalLayer);
REGISTER_LAYER_CLASS(FacesProposal);


}


