
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Collecting Dataset from Path
dataset = image_dataset_from_directory(
    "Path",
    shuffle=True,
    batch_size=32,
    image_size=(299, 299),
)

# Class Information
class_names = dataset.class_names
print("Class Names:", class_names)
print("Number of Classes:", len(class_names))

# Find the batch size
for images, labels in dataset.take(1):  # Take one batch from the dataset
    print("Batch Size:", images.shape[0])
    print("Image Size:", images.shape[1:])

    # Check the data types and shapes
    print("Image Data Type:", images.dtype)
    print("Label Data Type:", labels.dtype)
    print("Label Shape:", labels.shape)
    print("Labels in Batch:", labels.numpy())

# Total number of batches in the dataset
total_batches = len(dataset)
print("Total Number of Batches:", total_batches)

# Total number of images in the dataset
total_images = total_batches * images.shape[0]
print("Total Number of Images:", total_images)

# Data Split
def get_dataset_partisions_tf(ds, train_split=0.75, val_split=0.15, test_split=0.1, shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    dataset_size = len(ds)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_ds = ds.take(train_size)
    remaining_ds = ds.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    test_ds = remaining_ds.skip(val_size)

    return train_ds, val_ds, test_ds


train_data, val_data, test_data = get_dataset_partisions_tf(dataset)
print("Train Data: ",len(train_data), "Validation Data: ",len(val_data), "Test Data: ",len(test_data))