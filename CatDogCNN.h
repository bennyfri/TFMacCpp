
#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class CatDogCNN
{
private:
    Scope i_root; //graph for loading images into tensors
    const int image_side; //assuming quare picture
    const int image_channels; //RGB
    //load image vars
    Output file_name_var;
    Output image_tensor_var;
    //training and validating the CNN
    Scope t_root; //graph
    unique_ptr<ClientSession> t_session;
    //CNN vars
    Output input_batch_var;
    Output input_labels_var;
    Output drop_rate_var; //use real drop rate in training and 1 in validating
    Output skip_drop_var; //use 0 in trainig and 1 in validating
    Output out_classification;
    Output logits;
    //Network maps
    map<string, Output> m_vars;
    map<string, TensorShape> m_shapes;
    map<string, Output> m_assigns;
    //Loss variables
    vector<Operation> v_out_grads;
    Output out_loss_var;
public:
    CatDogCNN(int side, int channels):i_root(Scope::NewRootScope()), t_root(Scope::NewRootScope()), image_side(side), image_channels(channels) {}
    Status CreateGraphForImage(bool unstack);
    Status ReadTensorFromImageFile(string& file_name, Tensor& outTensor);
    Status ReadFileTensors(string& folder_name, vector<pair<string, float>> v_folder_label, vector<pair<Tensor, float>>& file_tensors);
    Status ReadBatches(string& folder_name, vector<pair<string, float>> v_folder_label, int batch_size, vector<Tensor>& image_batches, vector<Tensor>& label_batches);
    Input XavierInit(Scope scope, int in_chan, int out_chan, int filter_side = 0);
    Input AddConvLayer(string idx, Scope scope, int in_channels, int out_channels, int filter_side, Input input);
    Input AddDenseLayer(string idx, Scope scope, int in_units, int out_units, bool bActivation, Input input);
    Status CreateGraphForCNN(int filter_side);
    Status CreateOptimizationGraph(float learning_rate);
    Status Initialize();
    Status TrainCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results, float& loss);
    Status ValidateCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results);
    Status Predict(Tensor& image, int& result);
};

