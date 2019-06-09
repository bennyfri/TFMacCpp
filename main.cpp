
#include <chrono>
#include "CatDogCNN.h"
using namespace std;
using namespace chrono;
using namespace tensorflow;
using namespace tensorflow::ops;

int main(int argc, const char * argv[])
{
    int image_side = 150;
    int image_channels = 3;
    CatDogCNN model(image_side, image_channels);
    Status s = model.CreateGraphForImage(true);
    TF_CHECK_OK(s);

    string base_folder = "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small/train";
    int batch_size = 20;
    
    vector<Tensor> image_batches, label_batches, valid_images, valid_labels;
    //Label: cat=0, dog=1
    s = model.ReadBatches(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, image_batches, label_batches);
    TF_CHECK_OK(s);
    
    base_folder = "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small/validation";
    s = model.ReadBatches(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, valid_images, valid_labels);
    TF_CHECK_OK(s);

    //CNN model
    int filter_side = 3;
    s = model.CreateGraphForCNN(filter_side);
    TF_CHECK_OK(s);
    s = model.CreateOptimizationGraph(0.0001f);//input is learning rate
    TF_CHECK_OK(s);

    //Run inititialization
    s = model.Initialize();
    TF_CHECK_OK(s);
    
    size_t num_batches = image_batches.size();
    assert(num_batches == label_batches.size());
    size_t valid_batches = valid_images.size();
    assert(valid_batches == valid_labels.size());

    int num_epocs = 20;
    //Epoc / Step loops
    for(int epoc = 0; epoc < num_epocs; epoc++)
    {
        cout << "Epoc " << epoc+1 << "/" << num_epocs << ":";
        auto t1 = high_resolution_clock::now();
        float loss_sum = 0;
        float accuracy_sum = 0;
        for(int b = 0; b < num_batches; b++)
        {
            vector<float> results;
            float loss;
            s = model.TrainCNN(image_batches[b], label_batches[b], results, loss);
            loss_sum += loss;
            accuracy_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            cout << ".";
        }
        cout << endl << "Validation:";
        float validation_sum = 0;
        for(int c = 0; c < valid_batches; c++)
        {
            vector<float> results;
            s = model.ValidateCNN(valid_images[c], valid_labels[c], results);
            validation_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            cout << ".";

        }
        auto t2 = high_resolution_clock::now();
        cout << endl << "Time: " << duration_cast<seconds>(t2-t1).count() << " seconds ";
        cout << "Loss: " << loss_sum/num_batches << " Results accuracy: " << accuracy_sum/num_batches << " Validation accuracy: " << validation_sum/valid_batches << endl;
    }
    //testing the model
    s = model.CreateGraphForImage(false);//rebuild the model without unstacking
    TF_CHECK_OK(s);
    base_folder = "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small/test";
    vector<pair<Tensor, float>> all_files_tensors;
    s = model.ReadFileTensors(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
    TF_CHECK_OK(s);
    //test a few images
    int count_images = 20;
    int count_success = 0;
    for(int i = 0; i < count_images; i++)
    {
        pair<Tensor, float> p = all_files_tensors[i];
        int result;
        s = model.Predict(p.first, result);
        TF_CHECK_OK(s);
        cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << endl;
        if(result == (int)p.second)
            count_success++;
    }
    cout << "total successes: " << count_success << " out of " << count_images << endl;
    return 0;
}
