# Imports
# ==================================================

# Python Libraries
# --------------------------------------------------
import os
import sys
import pickle
import pydicom
import cv2 
import pandas            as pd
import numpy             as np 
import matplotlib.pyplot as plt

from glob                    import glob
from random                  import sample
from sklearn.model_selection import train_test_split
from skimage.transform       import resize

from sklearn.metrics import (
    roc_curve, 
    auc,
    precision_recall_curve, 
    average_precision_score
)


# Keras Libraries
# --------------------------------------------------
from keras.models                import model_from_json
from keras.preprocessing.image   import ImageDataGenerator
from keras.optimizers            import Adam
from keras.applications.vgg16    import VGG16
from keras.applications.resnet   import ResNet50 
from keras.applications.xception import Xception 
from keras                       import backend as K 

from keras.models import (
    Sequential, 
    Model
)

from keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateScheduler
)


# Custom Modules
# --------------------------------------------------
PATH = os.path.abspath(os.path.join(".."))

if PATH not in sys.path:
    sys.path.append(PATH)
    
import config



# Data Frame Methods
# ==================================================

def get_random_xray_foreground(data, rand):
    """
    """
                                     
    # Get Random X-ray Pixels
    # --------------------------------------------------
    xray = imread_random_xray(data, rand).ravel()


    # Remove Background
    # --------------------------------------------------
    xray_mask   = (xray > config.OTSU_THRESH)
    xray_pixels = xray[xray_mask].tolist()
                          
    return xray_pixels



# EDA Methods
# ==================================================                    
                
def plot_age_dist(data):
    """
    """
    
    # Get Mean and Standard Deviation of Patient Age
    # --------------------------------------------------
    pa_min  = round(np.min(data["Patient Age"]))
    pa_max  = round(np.max(data["Patient Age"]))
    pa_mean = round(np.mean(data["Patient Age"]))
    pa_std  = round(np.std( data["Patient Age"]))

    print(
        f"The Mean for Patient Age:               {pa_mean}\n"
        f"The Minimum for Patient Age:            {pa_min }\n"
        f"The maximum for Patient Age:            {pa_max }\n"
        f"The Standard Deviation for Patient Age: {pa_std}"
    )


    # Plot Distribution
    # --------------------------------------------------
    plt.figure(figsize = (20, 4))

    plt.bar(
        x      = data["Patient Age"].value_counts().index,
        height = data["Patient Age"].value_counts().values / data.shape[0]
    )

    plt.xlabel("Patient Age")
    plt.ylabel("Proportion of Patient IDs for Age")
    plt.title("Proportion of Patient IDs for Age")
    plt.grid(alpha = 0.25)


    # Add Mean and Standard Deviation
    # --------------------------------------------------
    plt.axvline(x = pa_mean,              color = "red",                   linewidth = 1.0)
    plt.axvline(x = pa_mean + pa_std,     color = "red", linestyle = "--", linewidth = 0.5)
    plt.axvline(x = pa_mean - pa_std,     color = "red", linestyle = "--", linewidth = 0.5)
    plt.axvline(x = pa_mean + pa_std * 2, color = "red", linestyle = "--", linewidth = 0.5)
    plt.axvline(x = pa_mean - pa_std * 2, color = "red", linestyle = "--", linewidth = 0.5)
    plt.axvline(x = pa_mean + pa_std * 3, color = "red", linestyle = "--", linewidth = 0.5)
    plt.axvline(x = pa_mean - pa_std * 3, color = "red", linestyle = "--", linewidth = 0.5)

    None
    
    
def plot_comparison_hist_grid(disease_df_1, disease_1, disease_df_2, disease_2, rows = 2, cols = 3):  
    """
    """
    
    fig, axs = plt.subplots(rows, cols, figsize = (20, 4 * rows))

    for idx, ax in enumerate(axs.flatten()):
        img_1 = get_random_xray_foreground(disease_df_1, idx + 42)
        img_2 = get_random_xray_foreground(disease_df_2, idx + 42)
        
        ax.hist(img_1, bins = 256, color = "dodgerblue", label = disease_1)
        ax.hist(img_2, bins = 256, color = "orange",     label = disease_2)
                          
        ax.title.set_text(f"Pixel Intensity Value Comparison - {disease_1} vs. {disease_2}")
                          
        ax.grid(alpha = 0.25)
        ax.legend()
                
    None
    
    
def plot_disease_age_dist(data, disease):
    """
    """

    age_df = data[
        (data["Patient Age"]    <= 100) &
        (data[config.LABEL_COL] == disease)
    ].groupby([
        config.UID, 
        "Patient Age"
    ])[config.IMG_IDX_COL].count().reset_index().rename(columns = {
        config.IMG_IDX_COL : "Count"
    })

    plot_age_dist(age_df)
    
    
def plot_disease_count_dist(data):
    """
    """
    
    plt.figure(figsize = (20, 4))

    plt.bar(
        x      = data["Disease Count"].value_counts().index,
        height = data["Disease Count"].value_counts().values / data["Disease Count"].shape[0]
    )

    plt.xlabel("Count of Diseases Per Patient")
    plt.ylabel("Proportion of Disease Count Per Patient")
    plt.title("Proportion of Disease Count Per Patient")
    plt.grid(alpha = 0.25)

    None
    
    
def plot_gender_dist(data):
    """
    """

    # Group Patients by Unique Gender and Unique Patient ID
    # NOTE: Possibility a Patient could Align with Different Gender 
    # Avoids Duplicagte Patient ID's Gender Skewing Distribution 
    # --------------------------------------------------
    gender_df = data.groupby([
        config.UID, 
        "Patient Gender"
    ])[config.IMG_IDX_COL].count().reset_index().rename(columns = {
        config.IMG_IDX_COL : "Count"
    })


    # Plot Distribution
    # --------------------------------------------------
    plt.figure(figsize = (10, 4))

    plt.bar(
        x      = gender_df["Patient Gender"].value_counts().index,
        height = gender_df["Patient Gender"].value_counts().values / gender_df.shape[0]
    )

    plt.xlabel("Patient Gender")
    plt.ylabel(f"Proportion of {config.UID}s for Gender")
    plt.title(f"Proportion of{config.UID}s for Gender")
    plt.grid(alpha = 0.25)

    None
                                              
    
def plot_pa_dist(data):
    """
    """

    plt.figure(figsize = (10, 4))

    plt.bar(
        x      = data["View Position"].value_counts().index,
        height = data["View Position"].value_counts().values / data.shape[0]
    )

    plt.xlabel("View Position")
    plt.ylabel("Proportion of Cases")
    plt.title("Proportion of Cases by View Position")
    plt.grid(alpha = 0.25)

    None 
    
    
    
# Image Methods
# ==================================================

def imread_random_xray(data, rand):
    """
    """
    
    return plt.imread(data.sample(random_state = rand)[config.IMG_PATH_COL].iloc[0])


def imshow_random_xray(data):
    """
    """
    
    plt.figure(figsize = (7, 7))

    plt.imshow(
        X    = imread_random_xray(data), 
        cmap = "bone"
    )

    None
    
    
def imshow_xray_grid(data):  
    """
    """
    
    fig, axs = plt.subplots(2, 5, figsize = (20, 8))

    for idx, ax in enumerate(axs.flatten()):
        img = data.sample(random_state = idx + 34)
        
        ax.imshow(
            X    = plt.imread(img[config.IMG_PATH_COL].iloc[0]), 
            cmap = "bone"
        )
        
        ax.title.set_text(f"{img[config.LABEL_COL].iloc[0]}")
                
    None
                          
                          
def imshow_xray_grid_labels(gen_data):
    """
    """
                          
    img, label = next(gen_data)
    fig, axs   = plt.subplots(4, 4, figsize = (16, 16))

    for (x, y, ax) in zip(img, label, axs.flatten()):
        ax.imshow(x[:, :, 0], cmap = "bone")

        if y == 1: ax.set_title(config.TARGET_DISEASE)
        else:      ax.set_title(f"No {config.TARGET_DISEASE}")

        ax.axis("off")



# Data Engineering Methods
# ==================================================

def add_disease_ind(data):
    """
    """

    # Get Unique Diseases Labels (All Cases & config.TARGET_DISEASE Cases)
    # --------------------------------------------------
    all_diseases = sorted(list(set(data[config.LABEL_COL].str.split("|").sum())))
    all_diseases.remove("No Finding")


    # Create Indicator Columns for Each Diseases Comorbid with config.TARGET_DISEASE
    # --------------------------------------------------
    for a_disease in all_diseases:
        data[a_disease] = data[config.LABEL_COL].str.contains(a_disease).astype(int)
        
    return data


def add_image_path(data):
    """
    """

    all_image_paths           = {os.path.basename(x) : x for x in glob(os.path.join('/data','images*', '*', '*.png'))}
    data[config.IMG_PATH_COL] = data[config.IMG_IDX_COL].map(all_image_paths.get)

    return data



# Data Split Methods
# ==================================================

def data_split(data):
    """
    """
    
    # Split Data by Dependent Variable
    # --------------------------------------------------
    train_data, valid_data = train_test_split(
        data,
        test_size    = config.VAL_DATA_RATIO,
        stratify     = data[config.DV],
        random_state = 42
    )
    
    
    # Balance Classes in Training Data
    # --------------------------------------------------
    t_inds = train_data[train_data[config.DV] == "1"].index.tolist()
    f_inds = train_data[train_data[config.DV] == "0"].index.tolist() 

    f_sample   = sample(f_inds, len(t_inds))     
    train_data = train_data.loc[t_inds + f_sample]
    train_data = train_data.sample(frac = 1)
    
    
    # Ensure Patient IDs Not Shared Between Train and Valid
    # --------------------------------------------------
    pid_train = list(pd.unique(train_data[config.UID]))
    pid_valid = list(pd.unique(valid_data[config.UID]))
    valid_ids = [a_pid for a_pid in pid_valid if a_pid not in pid_train]
    
    valid_data = valid_data[valid_data[config.UID].isin(valid_ids)]


    # Emulate Real Data in Validation Data
    # --------------------------------------------------
    t_inds = valid_data[valid_data[config.DV] == "1"].index.tolist()
    f_inds = valid_data[valid_data[config.DV] == "0"].index.tolist()
    
    f_sample   = sample(f_inds, 4 * len(t_inds))
    valid_data = valid_data.loc[t_inds + f_sample]
    valid_data = valid_data.sample(frac = 1)
    
    
    # Print Result
    # --------------------------------------------------
    print(
        f"Removing Train {config.UID}s from Valid\n"
        f"Cases Removed: {len(pid_valid) - len(valid_ids)}\n"
    )
    
    print(
        f"Train Data Image Count:   {train_data.shape[0]}\n"
        f"Train Data Patient Count: {len(list(pd.unique(train_data[config.UID])))}\n"
        f"Train Data Positive:      {round(train_data[train_data[config.DV] == '1'].shape[0] / train_data.shape[0], 2)}\n"
        f"Train Data Negative:      {round(train_data[train_data[config.DV] == '0'].shape[0] / train_data.shape[0], 2)}\n"
        f"\n"
        f"Validation Data Image Count:   {valid_data.shape[0]}\n"
        f"Validation Data Patient Count: {len(list(pd.unique(valid_data[config.UID])))}\n"
        f"Validation Data Positive:      {round(valid_data[valid_data[config.DV] == '1'].shape[0] / valid_data.shape[0], 2)}\n"
        f"Validation Data Negative:      {round(valid_data[valid_data[config.DV] == '0'].shape[0] / valid_data.shape[0], 2)}\n"
    )
    
    return (
        train_data, 
        valid_data
    )



# Data Augmentation Methods
# ==================================================

def data_augmentation(data, split):
    """
    """
    
    if split == "train":
        idg = ImageDataGenerator(
            rescale                = 1.0 / 255.0,
            horizontal_flip        = config.H_FLIP,
            vertical_flip          = config.V_FLIP, 
            height_shift_range     = config.H_SHIFT,
            width_shift_range      = config.W_SHIFT,
            rotation_range         = config.ROTATION,
            shear_range            = config.SHEAR,
            zoom_range             = config.ZOOM,
            brightness_range       = config.BRIGHTNESS,
            fill_mode              = config.FILL_MODE,
            cval                   = config.CVAL,
            preprocessing_function = imagenet_norm
        )
        
    elif split == "valid":
        idg = ImageDataGenerator(
            rescale                = 1.0 / 255.0,
            preprocessing_function = imagenet_norm
        )
    
    return idg


def imagenet_norm(img):
    """
    """
    
    return (img - config.IMAGENET_MU) / config.IMAGENET_SIGMA



# Data Flow Methods
# ==================================================

def data_gen(data, split, batch_size, image_size):
    """
    """
    
    gen = data_augmentation(data, split).flow_from_dataframe(
        dataframe    = data, 
        directory    = None, 
        x_col        = config.IMG_PATH_COL,
        y_col        = config.DV,
        class_mode   = config.CLASS_MODE,
        target_size  = image_size, 
        batch_size   = batch_size,
        seed         = 42
    )

    return gen



# Model Methods
# ==================================================

def build_model(model_name, tune_layers = []):
    """
    """
    
    model, output_layer = load_pretrained_model(model_name)
    model               = tune_pretrained_model(model, tune_layers, output_layer)
    model               = compile_pretrained_model(model)
    
    return model


def compile_pretrained_model(model):
    """
    """
        
    model.compile(
        optimizer = Adam(lr = config.LEARN_RATE), 
        loss      = config.LOSS, 
        metrics   = config.METRICS
    )
    
    return model


def load_model(model_name, model_weights, model_history):
    """
    """
    
    # Load Model
    # --------------------------------------------------
    model_json = open(model_name, "r")
    model      = model_json.read()
    model      = model_from_json(model)
    
    model_json.close()
    
    
    # Load Model Weights
    # --------------------------------------------------
    model.load_weights(model_weights)
    
    
    # Compile Loaded Model
    # --------------------------------------------------
    model.compile(
        loss      = config.LOSS, 
        optimizer = Adam(lr = config.LEARN_RATE), 
        metrics   = config.METRICS
    )
    
    
    # Load Model History
    # --------------------------------------------------
    with open(model_history, "rb") as f:
        history = pickle.load(f)
        
        
    return(
        model,
        history
    )


def learn_rate_schedule(epoch, lr):
    """
    """
    
    if epoch < config.LR_PATIENCE: return lr
    else:                          return lr * np.exp(-config.LR_FACTOR)


def load_pretrained_model(model_name):
    """
    """
    
    if model_name == "ResNet50":
        model        = ResNet50(weights = "imagenet")
        output_layer = config.RESNET50_OUTPUT_LAYER

        
    elif model_name == "VGG16":
        model        = VGG16(weights = "imagenet")
        output_layer = config.VGG16_OUTPUT_LAYER
        
        
    elif model_name == "Xception":
        model        = Xception(weights = "imagenet")
        output_layer = config.XCEPTION_OUTPUT_LAYER

        
    model = Model(
        inputs  = model.input, 
        outputs = [model.get_layer(output_layer).output]
    )
    
    return (
        model, 
        output_layer
    )


def predict_with_model(model, x):
    """
    """

    return model.predict(
        x       = x, 
        verbose = True
    )


def train_models(models, model_name, train_gen, valid_gen):
    """
    """
    
    valid_x, valid_y = valid_gen.next()

    for a_model in models: 
            
        # Define Callback 
        # --------------------------------------------------
        callback_list = [
            ModelCheckpoint(
                filepath          = f"{config.WEIGHT_PATH}/{model_name}/{a_model}{config.WEIGHT_EXT}", 
                monitor           = config.CALLBACK_MONITOR,
                mode              = config.CALLBACK_MODE,
                save_best_only    = True,
                save_weights_only = True,
                verbose           = 1
            ), 
            EarlyStopping(
                monitor  = config.CALLBACK_MONITOR, 
                mode     = config.CALLBACK_MODE, 
                patience = config.EARLY_STOP
            ),
            LearningRateScheduler(
                learn_rate_schedule
            )
        ]


        # Train Model
        # --------------------------------------------------
        print(
            f"\n--------------------------------------------------\n"
            f"Training {a_model}"
            f"\n--------------------------------------------------\n"
        )
            
        history = models[a_model].fit_generator(
            generator       = train_gen, 
            validation_data = (valid_x, valid_y), 
            epochs          = config.EPOCHS, 
            callbacks       = callback_list,
            verbose         = 1
        )
        
        
        # Save Model
        # --------------------------------------------------
        with open(f"{config.MODEL_PATH}/{model_name}/{a_model}{config.MODEL_EXT}", "w") as f:
            f.write(models[a_model].to_json())


        # Save History
        # --------------------------------------------------
        with open(f"{config.HISTORY_PATH}/{model_name}/{a_model}{config.HISTORY_EXT}", "wb") as f:
            pickle.dump(history.history, f)


def tune_pretrained_model(model, tune_layers, output_layer):
    """
    """
    
    output_idx = model.layers.index(model.get_layer(output_layer))
    
    for layer in model.layers[0:output_idx]:
        layer.trainable = False
        
    if tune_layers: architecture = [model] + tune_layers
    else:           architecture = [model]

    return Sequential(architecture)    
    
    
                          
# Evaluation Methods
# ==================================================
                          
def calculate_f1(precision, recall):
    """
    """

    return 2 * (precision * recall) / (precision + recall) if recall and precision else 0
                          
                         
def plot_model_auc(pred, true):
    """
    """
    
    fig = plt.figure(
        figsize      = (20, 5),
        tight_layout = True
    )
    
    for idx, a_model in enumerate(pred):
        fig.add_subplot(1, len(pred.keys()), idx + 1)
        
        pred_df = pd.DataFrame({
            "true" : true.tolist(),
            "pred" : pred[a_model].flatten()
        })
                          
        fpr, tpr, thresholds = roc_curve(pred_df["true"], pred_df["pred"], pos_label = 1)

        plt.plot(fpr, tpr, label = f"{config.TARGET_DISEASE} (AUC: {round(auc(fpr, tpr), 2)})")
        plt.plot([0, 1], [0, 1], color = "gray", lw = 1, linestyle = "--")

        plt.title(f"Model {idx + 1}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend()
        plt.grid(alpha = 0.25)

        None
    
        
def plot_model_f1_threshold(pred, true):
    """
    """
    
    best_thresh = {a_model : 0 for a_model in pred}
    
    fig = plt.figure(
        figsize      = (20, 5),
        tight_layout = True
    )
    
    for idx, a_model in enumerate(pred):        
        pred_df = pd.DataFrame({
            "true" : true.tolist(),
            "pred" : pred[a_model].flatten()
        })
        
        precision, recall, thresholds = precision_recall_curve(pred_df["true"], pred_df["pred"], pos_label = 1)
        
        f1_score             = [calculate_f1(precision[i], recall[i]) for i in range(len(thresholds))]
        best_idx             = np.argmax(f1_score)
        best_thresh[a_model] = round(thresholds[best_idx], 2)
        
        print(
            f"Model {idx + 1}\n"
            f"--------------------------------------------------\n"
            f"Threshold: {str(round(best_thresh[a_model], 2))}\n"
            f"\n"
            f"Precision: {str(round(precision[best_idx], 2))}\n"
            f"Recall:    {str(round(recall[best_idx],    2))}\n"
            f"F1-Score:  {str(round(f1_score[best_idx],  2))}\n"
            f"\n"
        )
        
        fig.add_subplot(1, len(pred.keys()), idx + 1)
        
        plt.plot(thresholds, f1_score)
    
        plt.title(f"Model {idx + 1}")
        plt.xlabel("Threshold")
        plt.ylabel("F1-Score")

        plt.grid(alpha = 0.25)

    None
    
    return best_thresh
                          
                          
def plot_model_history(history):
    """
    """
    
    fig = plt.figure(
        figsize      = (20, 5),
        tight_layout = True
    )
    
    for idx, a_model in enumerate(history):
        fig.add_subplot(1, len(history.keys()), idx + 1)

        plt.plot(history[a_model][config.CALLBACK_MONITOR],    label = config.CALLBACK_MONITOR.replace("_", " "))
        plt.plot(history[a_model]["loss"],                     label = "train loss")
        plt.plot(history[a_model][f"val_{config.METRICS[0]}"], label = f"val {config.METRICS[0]}")
        plt.plot(history[a_model][config.METRICS[0]],          label = f"train {config.METRICS[0]}")
        
        plt.title(f"Model {idx + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        
        plt.legend()
        plt.grid(alpha = 0.25)
        
        None 
                          
                          
def plot_model_prc(pred, true):
    """
    """
    
    fig = plt.figure(
        figsize      = (20, 5),
        tight_layout = True
    )
    
    for idx, a_model in enumerate(pred):
        fig.add_subplot(1, len(pred.keys()), idx + 1)
                          
        pred_df = pd.DataFrame({
            "true" : true.tolist(),
            "pred" : pred[a_model].flatten()
        })
                          
        precision, recall, thresholds = precision_recall_curve(pred_df["true"], pred_df["pred"], pos_label = 1)

        plt.plot(recall, precision, label = 
             f"{config.TARGET_DISEASE} (AP Score: "
             f"{round(average_precision_score(pred_df['true'], pred_df['pred']), 2)})"
        )

        plt.title(f"Model {idx + 1}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.legend()
        plt.grid(alpha = 0.25)

        None
    
    
def plot_pred_true_xray_grid(x, y, pred, thresh):
    """
    """
    
    best_model  = max(thresh, key = thresh.get)
    best_thresh = thresh[best_model]
    fig, axs    = plt.subplots(5, 5, figsize = (20, 20))

    for idx, (img, true, ax) in enumerate(zip(x[0:25], y[0:25], axs.flatten())):
        ax.imshow(img[:, :, 0], cmap = "bone")
        
        prob = pred[best_model].flatten()[idx]
        
        if true == 1: 
            if prob > best_thresh: ax.set_title("T1, P1")
            else:                  ax.set_title("T1, P0")

        else:
            if prob > best_thresh: ax.set_title("T0, P1")
            else:                  ax.set_title("T0, P0")

        ax.axis('off')

    
    
# Inference Methods
# ==================================================

def check_dicom(filename): 
    """
    """
    
    # Log Message
    # --------------------------------------------------
    print(f"Dicom File: {filename}")

    
    # Load Dicom
    # --------------------------------------------------
    ds = pydicom.dcmread(filename)     
    
    
    # Filter and Return
    # --------------------------------------------------    
    if ds.PatientPosition  in config.VALID_PATIENT_POSITIONS and \
       ds.Modality         in config.VALID_MODALITY          and \
       ds.BodyPartExamined in config.VALID_BODY_PART:
        return (
            ds.pixel_array, 
            ds.StudyDescription
        )
    
    else:
        print(
            f"{filename} is not suitable for model...\n"
            f"\n"
            f"Patient Position: {ds.PatientPosition}\n"
            f"Modality:         {ds.Modality}\n"
            f"Body Part:        {ds.BodyPartExamined}\n"
            f"--------------------------------------------------\n"
        )
        
        return(None, None)
    
    
def get_class_activation_mapping(model, img):
    """
    """
    
    # Normalize Image for Prediction
    # --------------------------------------------------
    norm_img = resize(img, (config.INPUT_IMG_SIZE[1], config.INPUT_IMG_SIZE[2]))
    norm_img = norm_img.reshape((1, config.INPUT_IMG_SIZE[1], config.INPUT_IMG_SIZE[2], 1))
    norm_img = np.repeat(norm_img, config.INPUT_IMG_SIZE[3], axis = 3)
    
    
    # Get Model Prediction
    # --------------------------------------------------
    preds = model.predict(norm_img)
    label = predict_image(
        model  = model, 
        img    = norm_img, 
        thresh = config.BEST_THRESH
    )
    
    
    # Get CAM
    # --------------------------------------------------
    final_conv_layer = model.layers[1]
    prediction       = model.predict(norm_img)
    disease          = model.output[:, np.argmax(preds[0])]
    grads            = K.gradients(disease, final_conv_layer.get_output_at(0))[0]
    pooled_grads     = K.mean(grads, axis=(0, 1, 2))
    
    iterate = K.function(
        [model.get_input_at(0)], 
        [
            pooled_grads,
            final_conv_layer.output[0]
        ]
    )
    
    pooled_grads_value, conv_layer_output_value = iterate([norm_img])
    
    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        
    # Get Heatmap from CAM
    # --------------------------------------------------
    heatmap  = np.mean(conv_layer_output_value, axis=-1)
    heatmap  = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap  = cv2.resize(heatmap, (norm_img.shape[1], norm_img.shape[2]), interpolation = cv2.INTER_LINEAR)
    heatmap  = np.uint8(255 * heatmap)
    heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
    
    
    # Normalize Image for Heatmap Overlay
    # --------------------------------------------------
    norm_img = resize(img, (config.INPUT_IMG_SIZE[1], config.INPUT_IMG_SIZE[2], 3))
    numer    = norm_img - np.min(norm_img)
    denom    = (norm_img.max() - norm_img.min()) + 1e-8
    norm_img = numer / denom
    norm_img = (norm_img * 255).astype("uint8")
    
    
    # Overlay Image with CAM Heatmap
    # --------------------------------------------------
    output = cv2.addWeighted(norm_img, 0.8, heatmap, 1 - 0.5, 0)
    
    return output 
    
    
def predict_image(model, img, thresh): 
    """
    """
    
    prob = model.predict(img)[0][0]
    
    if prob > thresh: 
        return config.TARGET_DISEASE, prob
    
    else:
        return f"No {config.TARGET_DISEASE}", prob
    
    
def preprocess_image(img, img_mean, img_std, img_size): 
    """
    """
    
    norm_img = img / 255
    norm_img = (norm_img - img_mean) / img_std
    
    norm_img = resize(norm_img, (img_size[1], img_size[2]))
    norm_img = norm_img.reshape((1, img_size[1], img_size[2], 1))
    norm_img = np.repeat(norm_img, img_size[3], axis = 3)
    
    return norm_img