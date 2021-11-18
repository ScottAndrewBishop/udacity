# Imports
# ==================================================

# Python Modules
# -------------------------
import os
import glob
import cv2
import signal
import requests
import math
import numpy             as np
import pandas            as pd
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

from contextlib import contextmanager


# Torch Modules
# -------------------------
import torch

from torchvision      import transforms
from torch.utils.data import(
    Dataset, 
    DataLoader
)


# Custom Modules
# -------------------------
import config



# I/O Classes
# ==================================================

class FacialKeypointsDataset(Dataset):
    """
    """

    def __init__(self, csv_file, root_dir, transform = None):
        """
        """
        
        self.keypoints_frame = pd.read_csv(csv_file)
        self.root_dir        = root_dir
        self.transform       = transform

        
    def __len__(self):
        """
        """
        
        return len(self.keypoints_frame)

    
    def __getitem__(self, idx):
        """
        """
        
        # Load Image
        # --------------------------------------------------
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_name = self.keypoints_frame.iloc[idx, 0]
        image      = mpimg.imread(
            os.path.join(self.root_dir, image_name)
        )
        
        
        # Remove Alpha Channel
        # --------------------------------------------------
        if(image.shape[2] == 4):
            image = image[:, :, 0:3]
        
        
        # Create Sample Object
        # --------------------------------------------------
        sample = {
            "image"     : image, 
            "keypoints" : self.keypoints_frame.iloc[idx, 1:].values.astype(float).reshape(-1, 2)
        }

        if self.transform:
            sample = self.transform(sample)

            
        return sample
    


# Transform Classes
# ==================================================

class Normalize(object):
    """
    """        

    def __call__(self, sample):
        """
        """
        
        # Extract Image and Keypoints
        # --------------------------------------------------
        image     = sample["image"]
        keypoints = sample["keypoints"]
        
        image_copy     = np.copy(image)
        keypoints_copy = np.copy(keypoints)

        
        # Convert to Grayscale
        # --------------------------------------------------
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
     
        # Normalize Image Color and Standardize Keypoints
        # --------------------------------------------------
        return {
            "image"     : image_copy / 255.0, 
            "keypoints" : (keypoints_copy - config.NORM_KP_MEAN) / config.NORM_KP_STD
        }


    
class Rescale(object):
    """
    """

    def __init__(self, output_size):
        """
        """
        
        assert isinstance(output_size, (int, tuple))
        
        self.output_size = output_size

        
    def __call__(self, sample):
        """
        """
        
        # Extract Image and Keypoints
        # --------------------------------------------------
        image     = sample["image"]
        keypoints = sample["keypoints"]

        
        # Calculate Rescale Ratio
        # --------------------------------------------------
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w: 
                new_h = self.output_size * h / w
                new_w = self.output_size
                
            else:    
                new_h = self.output_size
                new_w = self.output_size * w / h
                
        else:
            new_h, new_w = self.output_size

            
        # Rescale Image & Rescale Keypoints
        # --------------------------------------------------
        new_h = int(new_h)
        new_w = int(new_w)
        
        return {
            "image"     : cv2.resize(image, (new_w, new_h)), 
            "keypoints" : keypoints * [new_w / w, new_h / h]
        }

    

class RandomCrop(object):
    """
    """

    def __init__(self, output_size):
        """
        """
        
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            
        else:
            assert len(output_size) == 2
            
            self.output_size = output_size

            
    def __call__(self, sample):
        """
        """
        
        # Extract Image and Keypoints
        # --------------------------------------------------
        image     = sample["image"]
        keypoints = sample["keypoints"]

        
        # Perform Random Crop
        # --------------------------------------------------
        h, w         = image.shape[:2]
        new_h, new_w = self.output_size

        top  = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return {
            "image" :  image[
                top  : top  + new_h,
                left : left + new_w
            ], 
            "keypoints" : keypoints - [left, top]
        }


    
class ToTensor(object):
    """
    """

    def __call__(self, sample):
        """
        """
        
        # Extract Image and Keypoints
        # --------------------------------------------------
        image     = sample["image"]
        keypoints = sample["keypoints"]
         
            
        # Add Grayscale Color Channel
        # --------------------------------------------------
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
            
        # Swap Color Axis
        #     numpy image: h x w x c
        #     torch image: c x h x w
        # --------------------------------------------------
        image = image.transpose((2, 0, 1))
        
        return {
            "image"     : torch.from_numpy(image),
            "keypoints" : torch.from_numpy(keypoints)
        }
    
    

# Train Methods
# ==================================================

def net_sample_output(test_loader, net):
    """
    """
    
    for i, sample in enumerate(test_loader):
        
        # Get Image and Keypoints
        # --------------------------------------------------
        images    = sample["image"]
        images    = images.type(torch.FloatTensor)
        keypoints = sample["keypoints"]

        
        # Get Network Predictions
        # --------------------------------------------------
        output = net(images)
        output = output.view(output.size()[0], config.NUM_KEYPOINTS, -1)
        
        if i == 0:
            return (
                images, 
                output, 
                keypoints
            )
        
        
def train_net(net, train_loader, criterion, optimizer):
    """
    """

    net.train()

    for epoch in range(config.EPOCHS):
        running_loss = 0.0

        for batch_i, data in enumerate(train_loader):
            images = data["image"]
            images = images.type(torch.FloatTensor)
            
            keypoints = data["keypoints"]
            keypoints = keypoints.view(keypoints.size(0), -1)
            keypoints = keypoints.type(torch.FloatTensor)
            
            output = net(images)
            loss   = criterion(output, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_i % 10 == 9:
                print(
                    f"--------------------------------------------------\n"
                    f"Epoch:        {epoch   + 1}\n"
                    f"Batch:        {batch_i + 1}\n"
                    f"Average Loss: {round(running_loss / 10, 4)}"
                )
                
                running_loss = 0.0

    print("Finished Training")
    
    
    
# Predict Methods
# ==================================================

def predict(model, image):
    """
    """
    
    predict = model(image)
    predict = predict.view(config.NUM_KEYPOINTS, 2).data.numpy()
    predict = predict * config.NORM_KP_STD + config.NORM_KP_MEAN
    
    return predict



# Filter Methods
# ==================================================

def filter_sunglass_resize(image, keypoints, rad):
    """
    """
    
    h = int(abs(keypoints[config.TOP_NOSE_KP,        1] - keypoints[config.BOTTOM_NOSE_KP,    1]))
    w = int(abs(keypoints[config.RIGHT_BROW_EDGE_KP, 0] - keypoints[config.LEFT_BROW_EDGE_KP, 0]))
    
    w, h = (
        int(abs(np.sin(rad) * h) + abs(np.cos(rad) * w)),
        int(abs(np.sin(rad) * w) + abs(np.cos(rad) * h))
    )
    
    image = cv2.resize(
        src           = image, 
        dsize         = (w, h), 
        interpolation = cv2.INTER_CUBIC
    )
    
    return (
        image, 
        h, 
        w
    )

    
def filter_sunglass_rotation(image, keypoints):
    """
    """
    
    # Get Facial Keypoint Landmarks
    # --------------------------------------------------
    x1, y1 = keypoints[config.RIGHT_BROW_EDGE_KP].astype(int)
    x2, y2 = keypoints[config.LEFT_BROW_EDGE_KP ].astype(int)
    
    
    # Get Sunglass Shape
    # --------------------------------------------------
    h, w = image.shape[:2]
    
    
    # Calculate Rotation Matrix
    # --------------------------------------------------
    adj = abs(x1 - x2)
    opp = abs(y1 - y2)
    hyp = np.sqrt(adj**2 + opp**2)
    
    rad = math.asin(opp / hyp)
    deg = math.degrees(rad)
    
    M = cv2.getRotationMatrix2D(
        center = (w / 2, h / 2), 
        angle  = -deg, 
        scale  = 1
    )
    
    
    # Resize Image to Fit Rotation
    # --------------------------------------------------
    new_w, new_h = (
        abs(np.sin(rad) * h) + abs(np.cos(rad) * w),
        abs(np.sin(rad) * w) + abs(np.cos(rad) * h)
    )
    
    M[0, 2] += (new_w - w) / 2 
    M[1, 2] += (new_h - h) / 2

    
    # Rotate Sunglasses
    # --------------------------------------------------
    image = cv2.warpAffine(
        src   = image, 
        M     = M, 
        dsize = (int(new_w), int(new_h))
    )
    
    return (
        image, 
        rad
    )


def filter_sunglasses(image, keypoints):
    """
    """
    
    # Copy Image
    # --------------------------------------------------
    image_copy = np.copy(image)
    
    
    # Load Sunglasses
    # --------------------------------------------------
    sunglasses = cv2.imread(config.SUNGLASS_FILEPATH, cv2.IMREAD_UNCHANGED)
    x, y       = keypoints[config.RIGHT_BROW_EDGE_KP].astype(int)
    
    
    # Rotate Sunglasses
    # --------------------------------------------------
    sunglasses, rad = filter_sunglass_rotation(
        image     = sunglasses, 
        keypoints = keypoints
    )


    # Resize Rotated Sunglasses
    # --------------------------------------------------
    sunglasses, h, w = filter_sunglass_resize(
        image     = sunglasses, 
        keypoints = keypoints, 
        rad       = rad
    )


    # ROI
    # --------------------------------------------------
    roi_color = image_copy[
        y : y + h,
        x : x + w
    ]


    # Remove Transparent Sunglass Pixels
    # --------------------------------------------------
    idx = np.argwhere(sunglasses[:, :, 3] > 0)


    # Add Sunglasses to Image
    # --------------------------------------------------
    for i in range(3):
        roi_color[
            idx[:, 0],
            idx[:, 1],
            i
        ] = sunglasses[
            idx[:, 0],
            idx[:, 1],
            i
        ] 

    image_copy[
        y : y + h,
        x : x + w
    ] = roi_color


    # Visualize Sunglasses
    # --------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize = (20, 7))


    # Keypoints
    axs[0].imshow(image, cmap = "gray")
    axs[0].scatter(
        keypoints[:, 0], 
        keypoints[:, 1], 
        s      = 40, 
        marker = ".", 
        c      = "m"
    )
    
    axs[0].title.set_text("Predicted Keypoints")
    axs[0].axis('off')


    # Sunglasses
    axs[1].imshow(image_copy)
    axs[1].title.set_text("Sunglass Overlay")
    axs[1].axis("off")

    None
    
    
    
# Transform Methods
# ==================================================

def normalize_numpy(image):
    """
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255.0
    
    return image


def random_crop_numpy(image):
    """
    """
    
    h, w         = image.shape[:2]
    new_h, new_w = (config.CROP_SIZE, config.CROP_SIZE)

    if h - new_h > 0: top = np.random.randint(0, h - new_h)
    else:             top = 0

    if w - new_w > 0: left = np.random.randint(0, w - new_w)
    else:             left = 0

    return image[
        top  : top  + new_h,
        left : left + new_w
    ]
    
    
def rescale_numpy(image):
    """
    """
    
    h, w = image.shape[:2]
        
    if h > w: 
        new_h = config.CROP_SIZE * h / w
        new_w = config.CROP_SIZE

    else:    
        new_h = config.CROP_SIZE
        new_w = config.CROP_SIZE * w / h

    new_h = int(new_h)
    new_w = int(new_w)
    
    image = cv2.resize(image, (new_w, new_h))
    
    return image
    
    
def to_tensor_numpy(image):
    """
    """
    
    image = image.reshape(image.shape[0], image.shape[1], 1)   
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    image = image.type(torch.FloatTensor)
    
    return image
    
    
    
# Plot Methods
# ==================================================

def plot_all_keypoints(image, ax, pred_keypoints, true_keypoints = None):
    """
    """

    ax.imshow(image, cmap = "gray")
    
    ax.scatter(
        pred_keypoints[:, 0], 
        pred_keypoints[:, 1], 
        s      = 20, 
        marker = ".", 
        c      = "m"
    )
    
    if true_keypoints is not None:
        ax.scatter(
            true_keypoints[:, 0], 
            true_keypoints[:, 1], 
            s      = 20, 
            marker = ".", 
            c      = "g"
        )
        
        
def plot_feature_maps(image, weights):
    """
    """
    
    # Get Sample Image
    # --------------------------------------------------
    sample_image = image.data.numpy()
    sample_image = np.squeeze(np.transpose(sample_image, (1, 2, 0)))


    # Visualize Sample Image
    # --------------------------------------------------
    plt.figure(figsize = (5, 5))
    plt.imshow(sample_image, cmap = "gray")
    plt.title("Original Image")
    plt.axis("off")

    None


    # Get Weights of First Conv Layer
    # --------------------------------------------------
    weights = weights.data.numpy()


    # Visualize Filters
    # --------------------------------------------------
    rows     = round(len(weights) * 2 / 8)
    cols     = round(len(weights) * 2 / rows)
    fig, axs = plt.subplots(rows, cols, figsize = (4 * cols, 4 * rows))

    for idx, ax in enumerate(axs.flatten()):
        if idx % 2 == 0:
            filter_image = cv2.filter2D(
                src    = sample_image, 
                ddepth = -1, 
                kernel = weights[int(np.floor(idx / 2))][0]
            )

            ax.imshow(filter_image, cmap = "gray")
            ax.title.set_text(f"Index {int(np.floor(idx / 2))}")
            ax.axis("off")

        else:
            ax.imshow(weights[int(np.floor(idx / 2))][0], cmap = "gray")
            ax.title.set_text(f"Index {int(np.floor(idx / 2))}")
            ax.axis("off")  

    plt.tight_layout()

    None

        
def plot_model_error(test_images, pred_keypoints, true_keypoints = None, num_images = config.DATA_LOADER_BATCH_SIZE):
    """
    """
    
    rows     = round(num_images / 4)
    cols     = round(num_images / rows)
    fig, axs = plt.subplots(rows, cols, figsize = (20, 4 * rows))
    
    for idx, ax in enumerate(axs.flatten()):
        
        # Un-Transform Images
        # --------------------------------------------------
        image = test_images[idx].data
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))

        
        # Un-Transform Predicted Keypoints
        # --------------------------------------------------
        un_pred_keypoints = pred_keypoints[idx].data
        un_pred_keypoints = un_pred_keypoints.numpy()
        un_pred_keypoints = un_pred_keypoints * config.NORM_KP_STD + config.NORM_KP_MEAN
        
        
        # Un-Transform True Keypoints
        # --------------------------------------------------
        ground_truth_keypoints = None
        
        if true_keypoints is not None:
            ground_truth_keypoints = true_keypoints[idx]         
            ground_truth_keypoints = ground_truth_keypoints * config.NORM_KP_STD + config.NORM_KP_MEAN
        
        
        # Visualize Error
        # --------------------------------------------------
        plot_all_keypoints(
            image          = np.squeeze(image), 
            ax             = ax,
            pred_keypoints = un_pred_keypoints, 
            true_keypoints = ground_truth_keypoints
        )
            
        ax.axis("off")
        
    plt.tight_layout()
    
    None
    

def plot_random_faces_from_dataframe(keypoints_frame):
    """
    """
    
    fig, axs = plt.subplots(2, 4, figsize = (20, 10))
    
    for idx, ax in enumerate(axs.flatten()):
        image_idx  = np.random.randint(0, len(keypoints_frame))
        keypoints  = keypoints_frame.iloc[image_idx, 1:].values.astype('float').reshape(-1, 2)
        image_name = keypoints_frame.iloc[image_idx, 0]
        image      = mpimg.imread(os.path.join(config.TRAIN_DATA_FILEPATH, image_name))
        
        ax.imshow(image)
        ax.scatter(
            keypoints[:, 0], 
            keypoints[:, 1], 
            s      = 20, 
            marker = ".", 
            c      = "m"
        ) 
        
        ax.title.set_text(image_name)
        
    plt.tight_layout()

    None
    
    
def plot_random_faces_from_dict(face_dataset):
    """
    """
    
    fig, axs = plt.subplots(2, 4, figsize = (20, 10))
    
    for idx, ax in enumerate(axs.flatten()):
        image_idx  = np.random.randint(0, len(face_dataset))
        image      = face_dataset[image_idx]["image"]
        keypoints  = face_dataset[image_idx]["keypoints"]
        
        ax.imshow(image)
        ax.scatter(
            keypoints[:, 0], 
            keypoints[:, 1], 
            s      = 20, 
            marker = ".", 
            c      = "m"
        ) 
        
    plt.tight_layout()

    None
    
    
def plot_transform_test(face_dataset, rescale_size, crop_size, idx):
    """
    """
    
    # Define Transforms
    # --------------------------------------------------
    rescale  = Rescale(rescale_size)
    crop     = RandomCrop(crop_size)
    composed = transforms.Compose([
        Rescale(rescale_size),
        RandomCrop(crop_size)
    ])
    
    
    # View Transform Tests
    # --------------------------------------------------
    sample = face_dataset[idx]

    print(f"Original Image Shape: {sample['image'].shape}")

    fig = plt.figure(figsize = (20, 5))

    for i, tx in enumerate([rescale, crop, composed]):
        transformed_sample = tx(sample)

        ax = plt.subplot(1, 3, i + 1)
          
        ax.set_title(type(tx).__name__)
        ax.imshow(transformed_sample['image'])
        ax.scatter(
            transformed_sample['keypoints'][:, 0], 
            transformed_sample['keypoints'][:, 1], 
            s      = 20, 
            marker = ".", 
            c      = "m"
        ) 

    plt.tight_layout()

    None
    
    
# Udacity Methods
# ==================================================

def _request_handler(headers):
    """
    """
    
    def _handler(signum, frame):
        requests.request(
            method  = "POST", 
            url     = config.KEEP_ALIVE_URL, 
            headers = headers
        )
        
    return _handler          
      

@contextmanager
def active_session(delay = config.DELAY, interval = config.INTERVAL):
    """
    """
    
    token = requests.request(
        method  = "GET", 
        url     = config.TOKEN_URL, 
        headers = config.TOKEN_HEADERS
    ).text
    
    headers          = {"Authorization" : "STAR " + token}
    delay            = max(delay, config.MIN_DELAY)
    interval         = max(interval, config.MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        signal.signal(
            signalnum = signal.SIGALRM, 
            handler   = _request_handler(headers)
        )
        
        signal.setitimer(
            signal.ITIMER_REAL, 
            delay, 
            interval
        )
        
        yield
        
    finally:
        signal.signal(
            signalnum = signal.SIGALRM, 
            handler   = original_handler
        )
        
        signal.setitimer(
            signal.ITIMER_REAL,
            0
        )


def keep_awake(iterable, delay = config.DELAY, interval = config.INTERVAL):
    """
    """
    
    with active_session(delay, interval): 
        yield from iterable