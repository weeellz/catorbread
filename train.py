import argparse
import torch
from torch.optim import Adam

from datetime import datetime

from dataset import get_data
import model

classes = ('cat', 'bread')
steps = 0
running_loss = 0
train_losses, test_losses = [], []

cat_images = args.cat_images
bread_images = args.bread_images

train_loader, validation_loader = get_data(cat_images, bread_images)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # Create model, optimizer and loss function
    model = SimpleNet(num_classes=args.num_classes)

    #Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    start = datetime.now()
    print(f'Time: {start}')

    for epoch in range(args.epochs):
        start_ep = datetime.now()
        for data in train_loader:
            steps += 1
            print(f'Step {steps}')
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % args.print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for data in validation_loader:
                        inputs = data['image'].to(device)
                        labels = data['label'].to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(validation_loader))            
                print(f"Steps: {steps}.. "
                      f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/args.print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validation_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validation_loader):.3f}")
                running_loss = 0
                model.train()
                
        end_ep = datetime.now()
        print(f'Time for epoch: {end_ep - start_ep}')    

    end = datetime.now()
    print(f'Time: {end}')
    print(f'Time elapsed: {end - start}')

#torch.save(model, 'cnncatorbread.pth')

def main():
    parser = argparse.ArgumentParser(description='Training script of CatOrBread.')
    parser.add_argument('--cat_images', type=str, help='Path to cat images.')
    parser.add_argument('--bread_images', type=str, help='Path to bread images.')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=5, help='Total number of epochs.')
    parser.add_argument('--print_every', type=int, default=10, help='Print every # iterations.')
    parser.add_argument('--num_classes', type=int, default=2, help='Num classes.')
    #parser.add_argument('--type', choices=['CNN', 'MLP'], default='CNN', help='Type of Network')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()