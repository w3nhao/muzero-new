from x_transformers import TransformerWrapper, Decoder, Encoder
    
def generate_selective_copy_data(num_samples, sequence_length, vocabulary_size, pad_token=0):
    X = torch.full((num_samples, sequence_length), pad_token, dtype=torch.long)  # Initialize with padding tokens
    y = torch.zeros_like(X)
    
    for i in range(num_samples):
        seq = []
        remaining_length = sequence_length
        
        while remaining_length > 1:  # Ensure there's space for at least 1 info token
            # Randomly decide the number of padding tokens before the next info token (at least 1)
            num_pads = np.random.randint(1, remaining_length)
            # Ensure space for at least one info token
            num_pads = min(num_pads, remaining_length - 1)
            
            # Add padding tokens
            seq.extend([pad_token] * num_pads)
            remaining_length -= num_pads
            
            # Add an info token (ensure non-padding token)
            info_token = np.random.randint(1, vocabulary_size)
            seq.append(info_token)
            remaining_length -= 1

        # Convert sequence to tensor and assign to X
        X[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
        # Extract non-padding tokens and place them at the end of the corresponding y sequence
        non_padding_tokens = [token for token in seq if token != pad_token]
        y[i, -len(non_padding_tokens):] = torch.tensor(non_padding_tokens, dtype=torch.long)
        
    return X, y
    
if __name__ == "__main__":
    from tqdm import tqdm
    from torch import nn
    import torch.optim as optim
    import torch
    import os
    from pprint import pprint
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    from torch.utils.tensorboard import SummaryWriter
    
    sequence_length = 128
    vocabulary_size = 20
    
    current_path = "/data/wenhao/projects/20230126_muzero-general/models/"
    
    num_samples = 100000
    train_samples = int(num_samples * 0.9)
    test_samples = num_samples - train_samples

    # if exists generated data, load it
    if os.path.exists(current_path + "generated_data.pt"):
        X, y = torch.load(current_path + "generated_data.pt")
    else:
        # Generate some data
        X, y = generate_selective_copy_data(num_samples, sequence_length, vocabulary_size)
        torch.save((X, y), current_path + "generated_data.pt")
    
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define or import your model here
    # For example, using a simple Transformer model
    model = TransformerWrapper(
        num_tokens=vocabulary_size,
        max_seq_len=128,
        attn_layers=Decoder(
            dim=256,
            depth=2,
            heads=8,
            cross_attend=False,
            use_simple_rmsnorm = True # set to true to use for all layers
            
        )
    )
    
    pprint(model)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjust learning rate as needed
    epochs = 5  # Adjust number of epochs as needed
    batch_size = 64  # Adjust batch size as needed
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter()
    print("Creating tensorboard logs, run 'tensorboard --logdir=runs' in the terminal and go to http://localhost:6006/")

    global_step = 0  # Initialize a global step counter

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()  # Ensure the model is in training mode
        epoch_loss = 0
        total = 0
        correct = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * X_batch.size(0)  # Loss for the batch
            epoch_loss += batch_loss  # Accumulate epoch loss
            _, predicted = torch.max(output, -1)
            total += y_batch.size(0)
            correct += (predicted.view(-1) == y_batch.view(-1)).sum().item()

            writer.add_scalar("Loss/Train_step", batch_loss / X_batch.size(0), global_step)  # Record batch loss
            global_step += 1  # Increment global step count
            
        epoch_loss /= total
        epoch_acc = correct / total
        writer.add_scalar("Loss/Train_epoch", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_acc, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate the model on the test set
        model.eval()  # Ensure the model is in evaluation mode
        test_loss = 0
        total = 0
        correct = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
                test_loss += loss.item() * X_batch.size(0)  # Multiply by batch size to consider different last batch size

                _, predicted = torch.max(output, -1)

                # Create a mask for non-padding tokens in the target
                non_pad_mask = y_batch.view(-1) != 0  # Assumes 'pad_token' is 0
                correct_predictions = (predicted.view(-1) == y_batch.view(-1)) & non_pad_mask
                total += non_pad_mask.sum().item()
                correct += correct_predictions.sum().item()

        test_loss /= total
        test_acc = correct / total
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


    writer.close()
