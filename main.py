import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from network import NeuralNet
from PIL import Image

OLD_MODEL = True
TRAIN     = True
EPOCHS    = 1000


model = NeuralNet()
if OLD_MODEL:
    model.load_state_dict(torch.load('resources/model_weights.pth'))
    model.eval()


if TRAIN:
    # load data
    train_data = CustomDataset()
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    # load data


    # train
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        train_dataiter = iter(train_loader)
        for i, batch in enumerate(train_dataiter):
            x_batch, y_batch = batch

            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
    # train


    # save
    torch.save(model.state_dict(), 'resources/model_weights.pth')
    # save


# Generate
image = Image.new(mode="RGB", size=(24, 24))
pixels = image.load()
for x in range(image.size[0]):
    for y in range(image.size[1]):
        cord = torch.tensor([x/image.size[0], y/image.size[1]], dtype=torch.float32)
        r, g, b = int(model(cord)[0] * 255), int(model(cord)[1] * 255), int(model(cord)[2] * 255)
        pixels[x, y] = (r, g, b)
# Generate

image.save('resources/Result.jpg')
image.show()
