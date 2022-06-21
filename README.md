# Worm YOLOv3 Training Data Loader

A small package made to convert the manually annotated images from the [WormBot](https://github.com/JasonNPitt/wormbot) into the proper format for training in [Worm YOLOv3](https://github.com/paolobif/Worm-Yolo3).


-----------

## Getting started



```bash
$ pip install -r requirements.txt
```

In [data_loader.py](data_loader.py):

```
    TRAIN = True ## utilizes already labeled bounding-boxes
    CUT_SIZE = 416 ## cuts 416x416 (default)
    VAL = 0.2 ## proportion of the data that will be allocated to validation set
    DATA_CAP = 2000 ## number of image slices to cap the dataset size to
    EMPTY_RATIO = 0.5  ## ratio of images without worms to include in training.
    CSV_PATH = "/path/to/compiled/csv"
    IMAGE_PATH = "/path/to/directory/with/images"
    OUT_NAME = "train_folder" ## name for the training folder
```

**After specifying desired parameters:**
```bash
$ python3 data_loader.py
```

The generated folder can be to be used in [Worm YOLOv3](https://github.com/paolobif/Worm-Yolo3) for custom training.

**Should look as follows:**

    OUT_NAME/

    classes.names   OUT_NAME.data images   labels   train.txt   valid.txt


-------------


### CSV_PATH file format
Each row contains the image name, and the location for an individual worm. This one file should contain all the manual annotations for the entire training dataset.


***/path/to/compiled/csv.csv***

```

    image_name_1.png, x1, y1, w, h
    image_name_1.png, x1, y1, w, h
    image_name_2.png, x1, y1, w, h
    image_name_2.png, x1, y1, w, h
    image_name_2.png, x1, y1, w, h
    ....
```

### IMAGE_PATH
Is a folder containing the raw images for training.


***/path/to/directory/with/images/***

    image_name_1.png  image_name_2.png  image_name_3.png  image_name_4.png  ...


-----
