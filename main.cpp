
#include <chrono>
#include <iomanip>
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
    Status s;
    string proto_name = "/Users/bennyfriedman/Code/TF2example/TF2example/frozen/cnn.pb";

    bool augment_data = true;
    bool use_frozen = false;
    if(!use_frozen)
    {
        //continue with building and training the model
        s = model.CreateGraphForImage(true);
        TF_CHECK_OK(s);

        string base_folder = "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small/train";
        int batch_size = 20;
        
        if(augment_data)
        {
            float flip_ratio = 0.5f;
            float rotation_max_angles = 40.f;
            float scale_shift_factor = 0.2f;
            TF_CHECK_OK(model.CreateAugmentGraph(batch_size, image_side, flip_ratio, rotation_max_angles, scale_shift_factor));
        }
        
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
        s = model.CreateOptimizationGraph(0.0002f);//input is learning rate
        TF_CHECK_OK(s);

        //Run inititialization
        s = model.Initialize();
        TF_CHECK_OK(s);
        
        size_t num_batches = image_batches.size();
        assert(num_batches == label_batches.size());
        size_t valid_batches = valid_images.size();
        assert(valid_batches == valid_labels.size());

        SummaryWriterInterface *w1, *w2, *w3;
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs/loss/", "loss", Env::Default(), &w1));
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs/accu/", "loss", Env::Default(), &w2));
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs/valid/", "loss", Env::Default(), &w3));
        
        int num_epochs = 30;
        //Epoch / Step loops
        for(int epoch = 0; epoch < num_epochs; epoch++)
        {
            cout << "Epoch " << epoch+1 << "/" << num_epochs << ":";
            auto t1 = high_resolution_clock::now();
            float loss_sum = 0;
            float accuracy_sum = 0;
            for(int b = 0; b < num_batches; b++)
            {
                vector<float> results;
                float loss;
                if(augment_data)
                {
                    TF_CHECK_OK(model.WriteBatchToImageFiles(image_batches[b], "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small", "source"));
                    Tensor augmented;
                    TF_CHECK_OK(model.RandomAugmentBatch(image_batches[b], augmented));
                    TF_CHECK_OK(model.WriteBatchToImageFiles(augmented, "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small", "augmented"));
                    s = model.TrainCNN(augmented, label_batches[b], results, loss);
                }
                else
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
            Tensor t(DT_FLOAT, TensorShape({1}));
            t.scalar<float>()(0) = loss_sum/num_batches;
            TF_CHECK_OK(w1->WriteScalar(epoch, t, augment_data? "Augmented" : "Original"));
            t.scalar<float>()(0) = accuracy_sum/num_batches;
            TF_CHECK_OK(w2->WriteScalar(epoch, t, augment_data? "Augmented" : "Original"));
            t.scalar<float>()(0) = validation_sum/valid_batches;
            TF_CHECK_OK(w3->WriteScalar(epoch, t, augment_data? "Augmented" : "Original"));
        }
        //testing the model
        s = model.CreateGraphForImage(false);//rebuild the image loading model without unstacking
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
        
        s = model.FreezeSave(proto_name);
        TF_CHECK_OK(s);
    }
    else //use the frozen model
    {
        s = model.LoadSavedModel(proto_name);
        TF_CHECK_OK(s);
        //testing the model
        s = model.CreateGraphForImage(false);//rebuild the image loading model without unstacking
        TF_CHECK_OK(s);
        string base_folder = "/Users/bennyfriedman/Code/TF2example/TF2example/data/cats_and_dogs_small/test";
        vector<pair<Tensor, float>> all_files_tensors;
        s = model.ReadFileTensors(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
        TF_CHECK_OK(s);
        //test the images
        int count_success = 0;
        for(int i = 0; i < all_files_tensors.size(); i++)
        {
            pair<Tensor, float> p = all_files_tensors[i];
            int result;
            s = model.PredictFromFrozen(p.first, result);
            TF_CHECK_OK(s);
            if(i%10 == 0)
                cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << endl;
            if(result == (int)p.second)
                count_success++;
        }
        cout << "total successes: " << count_success << " out of " << all_files_tensors.size() << " which is " << setprecision(5) << (float)count_success / all_files_tensors.size() * 100 << "%" << endl;
    }
    return 0;
}
