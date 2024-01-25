# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
%matplotlib inline

import numpy as np
import sys, pickle, os, torch, cv2
# replace this path with path to GMIC
#sys.path.append("/path/to/GMIC")

from src.data_loading import loading
from src.modeling import gmic as gmic

from PIL import Image

# %% [markdown]
# ### You need to obtain the cropped images (bash run.sh) before using this demo.

# %% [markdown]
# Step 1: prepare the input data

# %%
with open("sample_data_vindr/data.pkl", "rb") as f:
    exam_list = pickle.load(f)

print(exam_list)
print(exam_list[0])
print(exam_list[0]["horizontal_flip"])
datum = exam_list[0]
view = "R-CC"
short_file_path = datum[view][0]
print(short_file_path)

# %% [markdown]
# Step 2: load the image

# %%
loaded_image = loading.load_image(
                    image_path=os.path.join("sample_data_vindr/cropped_images/", short_file_path + ".png"),
                    view=view, horizontal_flip=datum["horizontal_flip"])

print(type(loaded_image))
print(np.min(loaded_image))
print(np.max(loaded_image))
print(loaded_image.shape)

plt.imshow(loaded_image, cmap="gray")

# %% [markdown]
# Step 3: preprocess the image

# %%
loaded_image = loading.process_image(loaded_image, view, datum["best_center"][view][0])

# %% [markdown]
# Let's visualize a sample image

# %%
print(loaded_image.shape)
print(np.max(loaded_image))
plt.imshow(loaded_image, cmap="Greys_r")
plt.axis("off")
plt.savefig("mammoOriginal.png")
plt.show()

# %% [markdown]
# Step 4: load the model

# %%
parameters = {
        "device_type":"cpu",
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "percent_t": 0.02, 
        "post_processing_dim": 256,
        "num_classes": 2
    }

# %%
model = gmic.GMIC(parameters)

# %%
model.load_state_dict(torch.load("models/sample_model_5.p", map_location="cpu"), strict=False)

# %% [markdown]
# Step 5: inference

# %%
tensor_batch = torch.Tensor(np.expand_dims(np.expand_dims(loaded_image, 0), 0).copy())
output = model(tensor_batch)
pred_numpy = output.data.cpu().numpy()
benign_pred = pred_numpy[0,0]
malignant_pred = pred_numpy[0, 1]

# %%
print("benign prediction = {}".format(benign_pred))
print("malignant prediction = {}".format(malignant_pred))

# %% [markdown]
# Step 6: visualize the saliency map

# %%
saliency_maps = model.saliency_map.data.cpu().numpy()
print(saliency_maps.shape)
malignant_saliency_map = saliency_maps[0,1,:,:]
alphas = np.abs(np.linspace(0, 0.95, 259))
alpha_red = plt.cm.get_cmap('Reds')
plt.figure()
plt.imshow(loaded_image, cmap="Greys_r")
plt.imshow(cv2.resize(malignant_saliency_map, (1920, 2944)), alpha=0.3, cmap=alpha_red, clim=[0.0, 1.0])
plt.axis("off")
plt.savefig("mammoResult.png")
plt.show()

# %%


# %%



