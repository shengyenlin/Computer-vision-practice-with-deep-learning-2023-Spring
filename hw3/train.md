# Reproduce training results

Before training **any** model, please download the dataset provided by the TAs, and place it at the same directory level of `MIC` directory.
- train source model
```bash
python3 ./MIC/det/tools/train_net.py --mode source_only --hw_output_dir ./source_model
```
- train source model with MIC (adapted model)
  
Before training this model, run `bash download_fake_fog_train_json.sh`. You'll see a file called `train_fake.coco.json` in your current directory, please put this file into the same directory as `./hw3_dataest/fog/val.coco.json`

It's a fake training foggy dataset json file with random bounding bboxes and labels. It's needed for this package to train the model, but this fake json file won't be taken into the training process of the UDA model. Then, run the following code:
```bash 
python3 ./MIC/det/tools/train_net.py --mode source_with_da --hw_output_dir ./adapted_model
```

or run (the code above and the code below are the same.)
```bash
bash train.sh
```

- train source model with MIC (adapted model) with source model as initial weight
```bash 
bash hw3_download_source_model.sh
python3 ./MIC/det/tools/train_net.py --mode source_with_da --hw_output_dir ./adapted_model_from_source_model
```

After training, you will see several models in the `[--hw_output_dir]` directory, since I store checkpoint model every 1,000 iteration by default (the code will run 100,000 iteration). 

Therefore 0%-model corresponds to model without training (`model_0000000.pth` in the directory), 33%-model correponds to model after training 33,000 iterations (`model_0033000.pth`) in the directory, 66%-model correponds to model after training 66,000 iterations (`model_0066000.pth`), 100%-model correponds to model after training 100,000 iterations (`model_final.pth`).

**Warning: The training process may take over 20 hours**