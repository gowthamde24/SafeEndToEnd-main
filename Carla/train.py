import argparse
from turtle import forward
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
num_epochs = 9
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

losses_1 = []
losses_2 = []
losses_3 = []

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
        theta = float(Y[5])/100.
        v = float(Y[6])/100.
        vperp = float(Y[7])/100.

        Y1 = torch.tensor([steering])
        Y2 = torch.tensor([curvature/curvature_factor,x/x_factor,theta/theta_factor,cont_no,v,vperp])
        return X.float(), Y1.float(), Y2.float()

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

    # Image preprocessing, normalization for the pretrained resnet
    cropped_dataset = croppedDataset(ims=image_paths,gt_output_loc=None)
    
    # Build data loader
    train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, pin_memory=True)
    
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
    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (images, steerings, Y2) in enumerate(train_dl):
            # Set mini-batch dataset
            images = images.to(device)
            steerings = steerings.to(device)
            Y2 = Y2.to(device)
            
            # Forward, backward and optimize
            if TRAIN_STEERING :
                outputs = model(images,Y2[:,4:5],Y2[:,5:])
                loss = criterion(outputs, steerings)
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
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
            
            # Print log info
            if i % log_step == 0:
                if TRAIN_STEERING :
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item())) 
                    torch.save(model.state_dict(), os.path.join(
                        model_path, 'model-last.ckpt'))
                
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