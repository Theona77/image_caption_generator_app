## 🧩 Data Description

### Images Folder
Contains approximately **8000 images** in `.jpg` or `.png` format.  
Each image is used as input for the image captioning model.

### captions.txt
A CSV-like text file containing:

| Column  | Description                     |
|--------|---------------------------------|
| image  | Image file name (e.g., 0001.jpg) |
| caption | Human-written description of the image |

Format example:
image,caption
0001.jpg,a dog running on grass
0002.jpg,a child playing with a ball

### ProcessedData
A CSV file named "processed_captions" with text containing captions after preprocessing

## 📦 Data Source

Source of the dataset:
- Name: Flickr 8k Dataset
- Website/URL: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Provider: Kaggle

## ⚠️ Notes
- Images are resized to 224x224 before processing.
- Captions are tokenized and padded before training.

