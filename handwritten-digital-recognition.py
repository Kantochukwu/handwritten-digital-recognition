import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from streamlit_drawable_canvas import st_canvas


# CNN MODEL
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# DATASET TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@st.cache_resource
def load_or_train_model():
    model_path = "mnist_cnn.pth"
    model = CNN()

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        model.eval()
        return model

    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), model_path)

    model.eval()
    return model


model = load_or_train_model()


# DRAWING PREPROCESS
def process_drawing(image):
    if image is None:
        return None

    canvas = image[:, :, :3]

    gray = np.dot(canvas[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    _, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = np.argwhere(gray > 0)

    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = gray[y0:y1 + 1, x0:x1 + 1]

    h, w = cropped.shape
    size = max(h, w) + 20

    square = np.zeros((size, size), dtype=np.uint8)

    y_off = (size - h) // 2
    x_off = (size - w) // 2

    square[y_off:y_off + h, x_off:x_off + w] = cropped

    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized

    final_img = final_img.astype(np.float32) / 255.0
    final_img = (final_img - 0.5) / 0.5

    tensor = torch.tensor(
        final_img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor


def predict_digit(image):
    img = process_drawing(image)

    if img is None:
        return None

    model.eval()

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return pred.item()


# STREAMLIT UI
st.set_page_config(page_title="Draw a digit")

st.title("Draw a digit")

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False


canvas = st_canvas(
    fill_color="black",
    stroke_width=14,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.clear_canvas}"
)

col1, col2 = st.columns(2)

with col1:
    submit = st.button("Submit")

with col2:
    clear = st.button("Clear")


if submit:
    if canvas.image_data is not None:
        prediction = predict_digit(canvas.image_data)
        st.session_state.prediction = prediction


if clear:
    st.session_state.prediction = None
    st.session_state.clear_canvas = not st.session_state.clear_canvas
    st.rerun()


if st.session_state.prediction is not None:
    st.success(f"Prediction: {st.session_state.prediction}")
else:
    st.info("Draw a digit")