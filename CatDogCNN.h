
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
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/contrib/image/image_ops.h"
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
    //data augmentation
    Scope a_root;
    Output aug_tensor_input;
    Output aug_tensor_output;
    //training and validating the CNN
    Scope t_root; //graph
    unique_ptr<ClientSession> t_session;
    unique_ptr<Session> f_session;
    //CNN vars
    Output input_batch_var;
    string input_name = "input";
    Output input_labels_var;
    Output drop_rate_var; //use real drop rate in training and 1 in validating
    string drop_rate_name = "drop_rate";
    Output skip_drop_var; //use 0 in trainig and 1 in validating
    string skip_drop_name = "skip_drop";
    Output out_classification;
    string out_name = "output_classes";
    Output logits;
    //Network maps
    map<string, Output> m_vars;
    map<string, TensorShape> m_shapes;
    map<string, Output> m_assigns;
    //Loss variables
    vector<Output> v_weights_biases;
    vector<Operation> v_out_grads;
    Output out_loss_var;
    InputList MakeTransforms(int batch_size, Input a0, Input a1, Input a2, Input b0, Input b1, Input b2);
public:
    CatDogCNN(int side, int channels):i_root(Scope::NewRootScope()), t_root(Scope::NewRootScope()), a_root(Scope::NewRootScope()), image_side(side), image_channels(channels) {}
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
    Status FreezeSave(string& file_name);
    Status LoadSavedModel(string& file_name);
    Status PredictFromFrozen(Tensor& image, int& result);
    Status CreateAugmentGraph(int batch_size, int image_side, float flip_chances, float max_angles, float sscale_shift_factor);
    Status RandomAugmentBatch(Tensor& image_batch, Tensor& augmented_batch);
    Status WriteBatchToImageFiles(Tensor& image_batch, string folder_name, string image_name);
};

