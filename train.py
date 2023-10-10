from tqdm import tqdm

def train(model,optimizer,criterion,train,epochs,device):
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train)
            epoch_loss += loss / len(train)

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n"
        )