# import cv2
# import numpy as np
# import pygame
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
#
# # CNN MODEL
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.25)
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
#         x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 7x7
#         x = x.view(-1, 64 * 7 * 7)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
#
# # DATASET
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# trainset = torchvision.datasets.MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )
#
# testset = torchvision.datasets.MNIST(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transform
# )
#
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)
#
#
# # MODEL SETUP
# model = CNN()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # TRAINING
# epochs = 15
#
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#
#     for images, labels in trainloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")
#
#
# # TESTING
# correct = 0
# total = 0
#
# model.eval()
#
# with torch.no_grad():
#     for images, labels in testloader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f"Accuracy: {100 * correct / total:.2f}%")
#
#
# # DRAWING PREPROCESS
# def process_drawing(screen, canvas_size):
#     surface = pygame.surfarray.array3d(screen)
#     canvas = surface[:, :canvas_size, :]
#
#     gray = np.dot(canvas[..., :3], [0.299, 0.587, 0.114])
#     gray = np.transpose(gray, (1, 0)).astype(np.float32)
#
#     coords = np.argwhere(gray > 10)
#     if len(coords) == 0:
#         return None
#
#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0)
#
#     padding = 16
#     y0 = max(0, y0 - padding)
#     x0 = max(0, x0 - padding)
#     y1 = min(gray.shape[0], y1 + padding)
#     x1 = min(gray.shape[1], x1 + padding)
#
#     cropped = gray[y0:y1, x0:x1]
#
#     h, w = cropped.shape
#     size = max(h, w)
#
#     square = np.zeros((size, size), dtype=np.float32)
#
#     y_off = (size - h) // 6
#     x_off = (size - w) // 6
#
#     square[y_off:y_off + h, x_off:x_off + w] = cropped
#
#     resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
#     resized = resized / 255.0
#     resized = (resized - 0.5) / 0.5
#
#     tensor = torch.tensor(
#         resized,
#         dtype=torch.float32
#     ).unsqueeze(0).unsqueeze(0)
#
#     return tensor
#
#
# def predict_digit(screen, canvas_size):
#     img = process_drawing(screen, canvas_size)
#
#     if img is None:
#         return None
#
#     model.eval()
#
#     with torch.no_grad():
#         output = model(img)
#         _, pred = torch.max(output, 1)
#
#     return pred.item()
#
#
# # PYGAME DRAW UI
# def draw_digit():
#     pygame.init()
#
#     canvas_size = 280
#     panel_height = 80
#
#     width = canvas_size
#     height = canvas_size + panel_height
#
#     screen = pygame.display.set_mode((width, height))
#     pygame.display.set_caption("Draw a digit")
#
#     clock = pygame.time.Clock()
#     font = pygame.font.Font(None, 32)
#
#     drawing = False
#     prediction = None
#
#     submit_btn = pygame.Rect(20, canvas_size + 25, 100, 40)
#     clear_btn = pygame.Rect(160, canvas_size + 25, 100, 40)
#
#     def draw_buttons():
#         pygame.draw.rect(screen, (70, 70, 70), submit_btn, border_radius=6)
#         pygame.draw.rect(screen, (70, 70, 70), clear_btn, border_radius=6)
#
#         submit_text = font.render("Submit", True, (255, 255, 255))
#         clear_text = font.render("Clear", True, (255, 255, 255))
#
#         screen.blit(
#             submit_text,
#             (
#                 submit_btn.x + (submit_btn.width - submit_text.get_width()) // 2,
#                 submit_btn.y + 8
#             )
#         )
#
#         screen.blit(
#             clear_text,
#             (
#                 clear_btn.x + (clear_btn.width - clear_text.get_width()) // 2,
#                 clear_btn.y + 8
#             )
#         )
#
#     def refresh_ui():
#         screen.fill((30, 30, 30), (0, canvas_size, width, panel_height))
#
#         if prediction is not None:
#             text = font.render(
#                 f"Prediction: {prediction}",
#                 True,
#                 (0, 255, 0)
#             )
#         else:
#             text = font.render(
#                 "Draw a digit",
#                 True,
#                 (180, 180, 180)
#             )
#
#         screen.blit(text, (10, canvas_size + 2))
#         draw_buttons()
#         pygame.display.flip()
#
#     screen.fill((0, 0, 0))
#     refresh_ui()
#
#     while True:
#         for event in pygame.event.get():
#
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return
#
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 x, y = event.pos
#
#                 if y < canvas_size:
#                     drawing = True
#
#                 elif submit_btn.collidepoint(event.pos):
#                     prediction = predict_digit(screen, canvas_size)
#                     refresh_ui()
#
#                 elif clear_btn.collidepoint(event.pos):
#                     screen.fill((0, 0, 0), (0, 0, canvas_size, canvas_size))
#                     prediction = None
#                     refresh_ui()
#
#             if event.type == pygame.MOUSEBUTTONUP:
#                 drawing = False
#
#             if event.type == pygame.MOUSEMOTION and drawing:
#                 if event.pos[1] < canvas_size:
#                     pygame.draw.circle(
#                         screen,
#                         (255, 255, 255),
#                         event.pos,
#                         4
#                     )
#
#                     pygame.display.update(
#                         pygame.Rect(
#                             event.pos[0] - 5,
#                             event.pos[1] - 5,
#                             10,
#                             10
#                         )
#                     )
#
#         clock.tick(60)
#
#
# draw_digit()

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from streamlit_drawable_canvas import st_canvas


# ---------------------------
# CNN MODEL
# ---------------------------
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


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(
        torch.load("mnist_cnn.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


model = load_model()


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def process_drawing(image):
    gray = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    coords = np.argwhere(gray > 10)

    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    padding = 16

    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(gray.shape[0], y1 + padding)
    x1 = min(gray.shape[1], x1 + padding)

    cropped = gray[y0:y1, x0:x1]

    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.float32)

    y_off = (size - h) // 2
    x_off = (size - w) // 2

    square[y_off:y_off + h, x_off:x_off + w] = cropped

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    resized = resized / 255.0
    resized = (resized - 0.5) / 0.5

    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor


def predict_digit(image):
    img = process_drawing(image)

    if img is None:
        return None

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return pred.item()


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Handwritten Digit Recognition")

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click **Predict**.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("Predict")

with col2:
    clear = st.button("Clear")


if predict:
    if canvas_result.image_data is not None:
        prediction = predict_digit(canvas_result.image_data)

        if prediction is None:
            st.warning("Please draw a digit first.")
        else:
            st.success(f"Prediction: {prediction}")


if clear:
    st.rerun()