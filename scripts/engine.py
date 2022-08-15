from tqdm import tqdm 
import numpy as np
import logging
import torch

class Engine(object):
    def __init__(self, model, optimizer, criterion, Epochs=100, Device='cpu', earlyStop=False, save=False, name_model="model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = Epochs
        self.device = Device
        self.earlyStop = earlyStop
        self.save = save
        self.name_model = name_model
        
        self.trainLoss = np.inf
        self.valLoss = np.inf
        self.no_improve = 0

    def train(self, trainLoader, valLoader=None):
        for epoch in range(self.epochs):
            trainLoss = 0
            self.model.train()
            tk = tqdm(trainLoader, total=len(trainLoader))
            for img, score, geo in tk:
                
                img = img.to(device=self.device)
                score = score.to(device=self.device)
                geo = geo.to(device=self.device)

                out_score, out_geo = self.model(img)
                
                self.optimizer.zero_grad()

                loss = self.criterion(score, geo, out_score, out_geo)

                loss.backward()

                self.optimizer.step()

                tk.set_postfix({"Epoch":epoch+1, 'Training Loss':loss.item()})
                
                trainLoss += loss.item() * img.size(0)
            
            trainLoss = trainLoss / len(trainLoader.dataset)
            logging.info(f"Epoch: {epoch + 1} - Training Loss : {trainLoss}")

            if valLoader is not None:
                valLoss = self.evaluation(valLoader, epoch)
                logging.info(f"Epoch: {epoch + 1} - Validation Loss : {valLoss}")
                
                if epoch > 5 and self.valLoss > valLoss:
                    self.valLoss = valLoss
                    self.no_improve = 0
                
                else:
                    self.no_improve += 1
                
            # Get Checkpoint
            if epoch > 5 and self.save:
                if self.trainLoss > trainLoss and self.valLoss > valLoss:
                    # Update Minimum loss
                    self.trainLoss = trainLoss
                    self.valLoss = valLoss
                    
                    # Payload Checkpoint
                    checkpoint = {'state_dict': self.model.state_dict(),
                                 'optimizer':optimizer.state_dict()}
                    
                    # Save Checkpoint
                    torch.save(checkpoint, self.name_model)
                    logging.info(f"Saving the model in {self.model} checkpoint")
                    
                    # Reset for earlyStopping
                    self.no_improve = 0
                    
            # Early Stopping                    
            if epoch > 5 and self.earlyStop == self.no_improve:
                logging.error("Early Stopping")
                logging.info("Training Stopped")
                break
            
            else:
                continue          
                
    def evaluation(self, dataloader, epoch=None):
        total_loss = 0
        self.model.eval()
        tk = tqdm(dataloader, total=len(dataloader))
        for img, score, geo in tk:
            img = img.to(device=self.device)
            score = score.to(device=self.device)
            geo = geo.to(device=self.device)

            out_score, out_geo = self.model(img)

            loss = self.criterion(score, geo, out_score, out_geo)

            total_loss += loss.item() * img.size(0)

            if epoch is not None:
                tk.set_postfix({"Epoch" : epoch+1, "Validation Loss": loss.item()})
            else:
                tk.set_postfix({"Validation Loss ": loss.item()})

        return total_loss / len(dataloader.dataset)         
