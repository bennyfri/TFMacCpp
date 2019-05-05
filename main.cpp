
#include <iostream>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               vector<Tensor>* out_tensors,
                               bool writeGraph)
{
    auto root = Scope::NewRootScope();
    
    auto file_name_var = Placeholder(root.WithOpName("input"), DT_STRING);
    auto file_reader = ReadFile(root.WithOpName("file_readr"), file_name_var);

    if (!str_util::EndsWith(file_name, ".jpg"))
    {
        return errors::InvalidArgument("Image must be jpeg encoded");
    }
    const int wanted_channels = 3;
    auto image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(wanted_channels));
    
    auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    auto dims_expander = ExpandDims(root.WithOpName("dim"), float_caster, 0);
    auto resized = ResizeBilinear(root.WithOpName("size"), dims_expander, Const(root, {input_height, input_width}));
    auto d = Div(root.WithOpName("normalized"), Sub(root, resized, {input_mean}), {input_std});
    
    ClientSession session(root);
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {d}, out_tensors));

    if(writeGraph)
    {
        GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
        SummaryWriterInterface* w;
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs", ".img-graph", Env::Default(), &w));
        TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
    }
    return Status::OK();
}

Status WriteTensorToImageFile(const string& file_name, const int input_height,
                              const int input_width, const float input_mean,
                              const float input_std, vector<Tensor>& in_tensors)
{
    auto root = Scope::NewRootScope();
    auto un_normalized = Multiply(root.WithOpName("un_normalized"), Add(root, in_tensors[0], {input_mean}), {input_std});
    auto shaped = Reshape(root.WithOpName("reshape"), un_normalized, Const(root, {input_height, input_width, 3}));
    if(!root.ok())
        LOG(FATAL) << root.status().ToString();
    auto casted = Cast(root.WithOpName("cast"), shaped, DT_UINT8);
    auto image = EncodeJpeg(root.WithOpName("EncodeJpeg"), casted);

    vector<Tensor> out_tensors;
    ClientSession session(root);
    TF_CHECK_OK(session.Run({image}, &out_tensors));

    ofstream fs(file_name, ios::binary);
    fs << out_tensors[0].scalar<string>()();
    return Status::OK();
}

int main(int argc, const char * argv[])
{
    string image = "/Users/bennyfriedman/Code/TF2example/TF2example/data/grace_hopper.jpg";
    int32 input_width = 299;
    int32 input_height = 299;
    float input_mean = 0;
    float input_std = 255;
    vector<Tensor> resized_tensors;

    Status read_tensor_status = ReadTensorFromImageFile(image, input_height, input_width, input_mean, input_std, &resized_tensors, true);
    cout << resized_tensors[0].shape().DebugString() << endl;
    if (!read_tensor_status.ok())
    {
        LOG(ERROR) << read_tensor_status;
        return -1;
    }
    Status write_tensor_staus = WriteTensorToImageFile("/Users/bennyfriedman/Code/TF2example/TF2example/data/output.jpg", input_height, input_width, input_mean, input_std, resized_tensors);
    return 0;
}
