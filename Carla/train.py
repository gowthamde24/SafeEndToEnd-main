import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from PIL import ImageFile
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
image_size = 256
WIDTH = 800
HEIGHT = 600
RUN_NO = 0

TRAIN_STEERING = True
TRAIN_X_CURV_THETA = True
LOAD_PREVIOUS_MODEL = True


batch_size = 16
learning_rate = 1e-3
num_epochs = 50   # ceiling -- early stopping (EARLY_STOP_PATIENCE) will typically stop sooner
EARLY_STOP_PATIENCE = 8  # stop after this many epochs with no val-loss improvement
save_step = 10
log_step = 1
transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.CenterCrop(min(HEIGHT,WIDTH)),
        transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

curvature_factor = 0.005
x_factor = 1
theta_factor = 5

# "Accuracy" for this regression task = % of predictions within a physical-unit tolerance
# of ground truth (steering/curvature in their raw label units, x in meters, theta in degrees).
TOLERANCES = {
    'steering': 2.0,
    'curvature': 0.002,
    'x': 0.15,
    'theta': 5.0,
}

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
BLOCK_SIZE = 20    # consecutive frames per block; frames are saved every 5 sim-ticks, so
                   # nearby frames are near-duplicates of the same driving moment -- splitting
                   # by block (not by individual frame) keeps val/test a genuine holdout
                   # instead of leaking near-identical scenes across train/val/test.
SPLIT_SEED = 42

losses_0 = []  # steering, matching the [trained_loss, zero_baseline_loss] convention below
losses_1 = []
losses_2 = []
losses_3 = []


def split_dataset(image_paths, block_size=BLOCK_SIZE, val_fraction=VAL_FRACTION,
                   test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    'Splits sorted, sequential frames into train/val/test by contiguous block (see BLOCK_SIZE).'
    blocks = [image_paths[i:i + block_size] for i in range(0, len(image_paths), block_size)]
    rng = random.Random(seed)
    rng.shuffle(blocks)

    n_blocks = len(blocks)
    if n_blocks < 3:
        print(f'Warning: only {n_blocks} block(s) of data -- using all of it for training, no val/test split.')
        return image_paths, [], []

    n_val = min(max(1, round(n_blocks * val_fraction)), n_blocks - 2)
    n_test = min(max(1, round(n_blocks * test_fraction)), n_blocks - 1 - n_val)

    val_blocks = blocks[:n_val]
    test_blocks = blocks[n_val:n_val + n_test]
    train_blocks = blocks[n_val + n_test:]

    train_paths = [p for block in train_blocks for p in block]
    val_paths = [p for block in val_blocks for p in block]
    test_paths = [p for block in test_blocks for p in block]
    return train_paths, val_paths, test_paths

def count_within_tolerance(pred, target, tolerance):
    'Count of samples (in a batch of shape [N,1]) whose |pred-target| <= tolerance.'
    return int((torch.abs(pred - target) <= tolerance).sum().item())

class EndtoEnd(models.resnet.ResNet):
    def __init__(self):
        super(EndtoEnd, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.speed_feat_extractor = nn.Linear(2, 64)
        self.final_layer = nn.Linear(64+128, 1)
        
    def forward(self, x, v, vperp):
        speed_feat = torch.cat([torch.atan2(vperp,v),torch.sqrt(v**2+vperp**2)],dim=1)
        x1 = super(EndtoEnd,self).forward(x)
        x2 = self.speed_feat_extractor(speed_feat)
        x = torch.cat((x1,x2),axis=1)
        x = self.final_layer(x)
        return x

class croppedDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims, gt_output_loc):
        'Initialization'
        self.ims = ims
        self.transform = transform
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ims)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        image_path = self.ims[index]
        image = Image.open(image_path)
        X = self.transform(image)
        
        # Safely extract just the filename without the folder path
        base_name = os.path.basename(image_path)
        Y = base_name.split('.')[0].split('_')
        
        cont_no = float(Y[1])
        steering = float(Y[2])/100.
        curvature = float(Y[3])/10000.
        x = float(Y[4])/100.
        # Wrap into [-180, 180] degrees: a handful of collected frames have raw values
        # near +/-360 deg from an unwrapped heading-error computation at collection time
        # (fixed at the source in run_iter.py, but older run*_images/ frames still need this).
        theta = (((float(Y[5])/100.) + 180.) % 360.) - 180.
        v = float(Y[6])/100.
        vperp = float(Y[7])/100.

        Y1 = torch.tensor([steering])
        Y2 = torch.tensor([curvature/curvature_factor,x/x_factor,theta/theta_factor,cont_no,v,vperp])
        return X.float(), Y1.float(), Y2.float()

def evaluate(model, model_safety_1, model_safety_2, model_safety_3, data_loader):
    """Runs a no-grad pass over data_loader.
    Returns (losses, accuracies), each [steering, curvature, x, theta]; accuracies are
    percent of samples within TOLERANCES of ground truth, in physical (unscaled) units."""
    criterion = nn.MSELoss()
    for m in (model, model_safety_1, model_safety_2, model_safety_3):
        m.eval()
    loss_totals = [0.0, 0.0, 0.0, 0.0]
    correct_counts = [0, 0, 0, 0]
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for images, steerings, Y2 in data_loader:
            images, steerings, Y2 = images.to(device), steerings.to(device), Y2.to(device)
            outputs = model(images, Y2[:,4:5], Y2[:,5:])
            outputs_1 = model_safety_1(images)
            outputs_2 = model_safety_2(images)
            outputs_3 = model_safety_3(images)

            loss_totals[0] += criterion(outputs, steerings).item()
            loss_totals[1] += criterion(outputs_1, Y2[:,0:1]).item()
            loss_totals[2] += criterion(outputs_2, Y2[:,1:2]).item()
            loss_totals[3] += criterion(outputs_3, Y2[:,2:3]).item()

            correct_counts[0] += count_within_tolerance(outputs, steerings, TOLERANCES['steering'])
            correct_counts[1] += count_within_tolerance(outputs_1*curvature_factor, Y2[:,0:1]*curvature_factor, TOLERANCES['curvature'])
            correct_counts[2] += count_within_tolerance(outputs_2*x_factor, Y2[:,1:2]*x_factor, TOLERANCES['x'])
            correct_counts[3] += count_within_tolerance(outputs_3*theta_factor, Y2[:,2:3]*theta_factor, TOLERANCES['theta'])

            n_samples += images.size(0)
            n_batches += 1
    for m in (model, model_safety_1, model_safety_2, model_safety_3):
        m.train()
    losses = [t / n_batches for t in loss_totals]
    accuracies = [100.0 * c / n_samples for c in correct_counts]
    return losses, accuracies

def main(args):
    global RUN_NO, IMAGE_FOLDER, model_path, model_path_prev, LOAD_PREVIOUS_MODEL
    if args.run_no!=-1 :
        RUN_NO = args.run_no
    IMAGE_FOLDER = 'run'+str(RUN_NO)+'_images'
    model_path = 'saved_models_iter'+str(RUN_NO) 
    model_path_prev = 'saved_models_iter'+str(RUN_NO-1)
    if RUN_NO==0 :
        LOAD_PREVIOUS_MODEL = False
        
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    image_paths = glob.glob(IMAGE_FOLDER+'/*.png')
    image_paths.sort()

    train_paths, val_paths, test_paths = split_dataset(image_paths)
    print(f'Dataset split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test frames')

    # Image preprocessing, normalization for the pretrained resnet
    cropped_dataset = croppedDataset(ims=train_paths,gt_output_loc=None)

    # Build data loaders
    train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(croppedDataset(ims=val_paths,gt_output_loc=None), batch_size,
                         shuffle=False, pin_memory=True) if val_paths else None
    test_dl = DataLoader(croppedDataset(ims=test_paths,gt_output_loc=None), batch_size,
                          shuffle=False, pin_memory=True) if test_paths else None
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    model = EndtoEnd()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(model.fc.in_features, 128))
    model = model.to(device)
    print(model)
    
    model_safety_1 = models.resnet18()
    model_safety_1.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_1 = model_safety_1.to(device)
    
    model_safety_2 = models.resnet18()
    model_safety_2.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_2 = model_safety_2.to(device)
    
    model_safety_3 = models.resnet18()
    model_safety_3.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_3 = model_safety_3.to(device)
    
    if LOAD_PREVIOUS_MODEL :
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last.ckpt'))
        model.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-1.ckpt'))
        model_safety_1.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-2.ckpt'))
        model_safety_2.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-3.ckpt'))
        model_safety_3.load_state_dict(st_dict)
        
    params = list(model.parameters()) 
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    params_safety_1 = list(model_safety_1.parameters()) 
    optimizer_safety_1 = torch.optim.Adam(params_safety_1, lr=learning_rate)
    
    params_safety_2 = list(model_safety_2.parameters()) 
    optimizer_safety_2 = torch.optim.Adam(params_safety_2, lr=learning_rate)
    
    params_safety_3 = list(model_safety_3.parameters()) 
    optimizer_safety_3 = torch.optim.Adam(params_safety_3, lr=learning_rate)
    
    # Train the models
    total_step = len(train_dl)
    val_loss_history = []
    train_val_accuracy_history = []
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_correct = [0, 0, 0, 0]
        train_samples = 0
        for i, (images, steerings, Y2) in enumerate(train_dl):
            # Set mini-batch dataset
            images = images.to(device)
            steerings = steerings.to(device)
            Y2 = Y2.to(device)

            # Forward, backward and optimize
            if TRAIN_STEERING :
                outputs = model(images,Y2[:,4:5],Y2[:,5:])
                loss = criterion(outputs, steerings)
                loss_zero = criterion(outputs*0, steerings)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                train_correct[0] += count_within_tolerance(outputs, steerings, TOLERANCES['steering'])

            if TRAIN_X_CURV_THETA :
                outputs_safety_1 = model_safety_1(images)
                loss_safety_1 = criterion(outputs_safety_1, Y2[:,0:1])
                loss_safety_1_zero = criterion(outputs_safety_1*0, Y2[:,0:1])

                outputs_safety_2 = model_safety_2(images)
                loss_safety_2 = criterion(outputs_safety_2, Y2[:,1:2])
                loss_safety_2_zero = criterion(outputs_safety_2*0, Y2[:,1:2])

                outputs_safety_3 = model_safety_3(images)
                loss_safety_3 = criterion(outputs_safety_3, Y2[:,2:3])
                loss_safety_3_zero = criterion(outputs_safety_3*0, Y2[:,2:3])

                model_safety_1.zero_grad()
                model_safety_2.zero_grad()
                model_safety_3.zero_grad()

                loss_safety_1.backward()
                optimizer_safety_1.step()

                loss_safety_2.backward()
                optimizer_safety_2.step()

                loss_safety_3.backward()
                optimizer_safety_3.step()

                train_correct[1] += count_within_tolerance(outputs_safety_1*curvature_factor, Y2[:,0:1]*curvature_factor, TOLERANCES['curvature'])
                train_correct[2] += count_within_tolerance(outputs_safety_2*x_factor, Y2[:,1:2]*x_factor, TOLERANCES['x'])
                train_correct[3] += count_within_tolerance(outputs_safety_3*theta_factor, Y2[:,2:3]*theta_factor, TOLERANCES['theta'])
            train_samples += images.size(0)

            # Print log info
            if i % log_step == 0:
                if TRAIN_STEERING :
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item()))
                    torch.save(model.state_dict(), os.path.join(
                        model_path, 'model-last.ckpt'))
                    losses_0.append([loss.item(),loss_zero.item()])
                    np.savetxt('losses_0.csv', np.array(losses_0))

                if TRAIN_X_CURV_THETA : 
                    print('Safety curvature : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_1.item())) 
                    torch.save(model_safety_1.state_dict(), os.path.join(
                        model_path, 'model-last-safety-1.ckpt'))
                    losses_1.append([loss_safety_1.item(),loss_safety_1_zero.item()])

                    print('Safety X : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_2.item())) 
                    torch.save(model_safety_2.state_dict(), os.path.join(
                        model_path, 'model-last-safety-2.ckpt'))
                    losses_2.append([loss_safety_2.item(),loss_safety_2_zero.item()])
                    
                    print('Safety theta : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_3.item())) 
                    torch.save(model_safety_3.state_dict(), os.path.join(
                        model_path, 'model-last-safety-3.ckpt'))
                    losses_3.append([loss_safety_3.item(),loss_safety_3_zero.item()])
                    np.savetxt('losses_1.csv', np.array(losses_1))
                    np.savetxt('losses_2.csv', np.array(losses_2))
                    np.savetxt('losses_3.csv', np.array(losses_3))

        train_accuracies = [100.0 * c / train_samples for c in train_correct]

        if val_dl is not None:
            val_losses, val_accuracies = evaluate(model, model_safety_1, model_safety_2, model_safety_3, val_dl)
            print('Validation - Epoch [{}/{}]: loss steering={:.4f} curvature={:.4f} x={:.4f} theta={:.4f}'
                  .format(epoch, num_epochs, *val_losses))
            print('Validation - Epoch [{}/{}]: acc%  steering={:.1f} curvature={:.1f} x={:.1f} theta={:.1f}  '
                  '(train acc%: steering={:.1f} curvature={:.1f} x={:.1f} theta={:.1f})'
                  .format(epoch, num_epochs, *val_accuracies, *train_accuracies))
            val_loss_history.append(val_losses)
            np.savetxt(f'val_losses_run{RUN_NO}.csv', np.array(val_loss_history), delimiter=',',
                        header='steering,curvature,x,theta', comments='')
            train_val_accuracy_history.append(train_accuracies + val_accuracies)
            np.savetxt(f'train_val_accuracy_run{RUN_NO}.csv', np.array(train_val_accuracy_history), delimiter=',',
                        header='train_steering,train_curvature,train_x,train_theta,'
                               'val_steering,val_curvature,val_x,val_theta', comments='')

            combined_val_loss = sum(val_losses)
            if combined_val_loss < best_val_loss:
                best_val_loss = combined_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(model_path, 'model-best.ckpt'))
                torch.save(model_safety_1.state_dict(), os.path.join(model_path, 'model-best-safety-1.ckpt'))
                torch.save(model_safety_2.state_dict(), os.path.join(model_path, 'model-best-safety-2.ckpt'))
                torch.save(model_safety_3.state_dict(), os.path.join(model_path, 'model-best-safety-3.ckpt'))
                print(f'New best val loss {best_val_loss:.4f} at epoch {epoch} -- saved model-best*.ckpt')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f'Early stopping: no val improvement for {EARLY_STOP_PATIENCE} epochs '
                          f'(best was epoch {best_epoch}, val loss {best_val_loss:.4f})')
                    break

    if test_dl is not None:
        if best_epoch >= 0:
            # Evaluate the best-val checkpoint (what run_iter.py will actually load), not
            # whatever the last epoch happened to leave in memory.
            model.load_state_dict(torch.load(os.path.join(model_path, 'model-best.ckpt')))
            model_safety_1.load_state_dict(torch.load(os.path.join(model_path, 'model-best-safety-1.ckpt')))
            model_safety_2.load_state_dict(torch.load(os.path.join(model_path, 'model-best-safety-2.ckpt')))
            model_safety_3.load_state_dict(torch.load(os.path.join(model_path, 'model-best-safety-3.ckpt')))
            print(f'Loaded best checkpoint (epoch {best_epoch}) for final test evaluation.')
        test_losses, test_accuracies = evaluate(model, model_safety_1, model_safety_2, model_safety_3, test_dl)
        print('Final test set evaluation: loss steering={:.4f} curvature={:.4f} x={:.4f} theta={:.4f}'
              .format(*test_losses))
        print('Final test set evaluation: acc%  steering={:.1f} curvature={:.1f} x={:.1f} theta={:.1f}'
              .format(*test_accuracies))
        with open(f'test_metrics_run{RUN_NO}.csv', 'w') as f:
            f.write('steering_loss,curvature_loss,x_loss,theta_loss,'
                    'steering_acc,curvature_acc,x_acc,theta_acc\n')
            f.write(','.join(str(v) for v in test_losses + test_accuracies) + '\n')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-r', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
    args = argparser.parse_args()
    
    main(args)