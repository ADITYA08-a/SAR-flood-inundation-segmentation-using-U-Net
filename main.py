import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.filters import gaussian
import segmentation_models_pytorch as smp
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# --- 1. Fixed Preprocessing functions ---
def apply_speckle_filter(img, sigma=1.0):
    return gaussian(img, sigma=sigma, preserve_range=True)

def normalize_sar(img, min_db=-30.0, max_db=0.0):
    """
    Smart normalization: Checks if data is already in Decibels (negative values)
    or linear (positive values) before scaling to [0, 1].
    """
    # If the average pixel value is positive, it's linear. Convert to dB.
    if np.nanmean(img) > 0:
        img = 10.0 * np.log10(np.clip(img, 1e-5, None))
    
    # Clip to standard Sentinel-1 backscatter range and normalize to [0,1]
    img = np.clip(img, min_db, max_db)
    return (img - min_db) / (max_db - min_db)

class FloodDataset(Dataset):
    def __init__(self, sar_dir, mask_dir, chip_ids):
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.chip_ids = chip_ids
    
    def __len__(self):
        return len(self.chip_ids)
    
    def __getitem__(self, index):
        chip_id = self.chip_ids[index]

        sar_path = os.path.join(self.sar_dir, chip_id)
        mask_filename = chip_id.replace("S1Hand", "LabelHand")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        with rasterio.open(sar_path) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        vv = np.nan_to_num(vv, nan=-30.0, posinf=-30.0, neginf=-30.0)
        vh = np.nan_to_num(vh, nan=-30.0, posinf=-30.0, neginf=-30.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        
        mask[mask <= 0] = 0.0
        mask[mask > 0] = 1.0 

        vv_filtered = apply_speckle_filter(vv)
        vh_filtered = apply_speckle_filter(vh)

        # Use the new smart normalization
        vv_norm = normalize_sar(vv_filtered)
        vh_norm = normalize_sar(vh_filtered)
        ratio_norm = np.clip(vv_norm / (vh_norm + 1e-5), 0.0, 1.0)

        sar_array = np.stack([vv_norm, vh_norm, ratio_norm], axis=0)
        sar_array = np.nan_to_num(sar_array, nan=0.0, posinf=0.0, neginf=0.0)

        sar_tensor = torch.from_numpy(sar_array).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return sar_tensor, mask_tensor

# --- 2. Train Pipeline ---
def train_pipeline(sar_dir, mask_dir, chip_ids, device):
    train_dataset = FloodDataset(sar_dir, mask_dir, chip_ids)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # CRITICAL FIX: Add pos_weight to penalize the model for missing flood pixels
    # Since water is rare in the imagery, we tell the loss function that water is 5x more important
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    epochs = 25
    print(f"\n[TRAINING] Starting Training Sequence with {len(chip_ids)} images...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for sar_inputs, target_masks in train_loader:
            sar_inputs = sar_inputs.to(device)
            target_masks = target_masks.to(device)

            optimizer.zero_grad()
            logits = model(sar_inputs)
            loss = criterion(logits, target_masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * sar_inputs.size(0)
        
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_loss:.5f}")
        
    print("[TRAINING] Complete. Saving weights to 'flood_unet.pth'...")
    torch.save(model.state_dict(), "flood_unet.pth")
    return model

# --- 3. Testing and Visualization Pipeline ---
def test_and_visualize_pipeline(model, sar_dir, mask_dir, test_chip_ids, device):
    print(f"\n[TESTING] Evaluating model on {len(test_chip_ids)} test images...")
    test_dataset = FloodDataset(sar_dir, mask_dir, test_chip_ids)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    
    test_loss = 0.0
    visualized = False

    with torch.no_grad():
        for batch_idx, (sar_inputs, target_masks) in enumerate(test_loader):
            sar_inputs = sar_inputs.to(device)
            target_masks = target_masks.to(device)
            
            logits = model(sar_inputs)
            loss = criterion(logits, target_masks)
            test_loss += loss.item() * sar_inputs.size(0)
            
            if not visualized:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                sample_sar = sar_inputs[0].cpu().numpy()
                sample_gt = target_masks[0, 0].cpu().numpy()
                sample_pred = preds[0, 0].cpu().numpy()
                
                # CRITICAL FIX: Contrast stretching so the SAR image isn't too dark to see
                def stretch_contrast(band):
                    p2, p98 = np.percentile(band, (2, 98))
                    return np.clip((band - p2) / (p98 - p2 + 1e-5), 0, 1)

                rgb_sar = np.stack([stretch_contrast(sample_sar[0]), 
                                    stretch_contrast(sample_sar[1]), 
                                    stretch_contrast(sample_sar[2])], axis=-1)
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.title("Raw SAR Input (Contrast Stretched)")
                plt.imshow(rgb_sar)
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                # Added vmin/vmax to ensure strict mapping of 0=white, 1=blue
                plt.imshow(sample_gt, cmap="Blues", vmin=0, vmax=1) 
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title("Model Prediction Mask")
                plt.imshow(sample_pred, cmap="Blues", vmin=0, vmax=1)
                plt.axis('off')
                
                output_image_path = "test_prediction_sample.png"
                plt.savefig(output_image_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"[VISUALIZATION] Saved visual comparison window to: '{os.path.abspath(output_image_path)}'")
                visualized = True

    avg_test_loss = test_loss / len(test_dataset)
    print(f"[TESTING] Evaluation Complete | Average Test Loss: {avg_test_loss:.5f}")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_DIR = os.path.join("archive(1)", "v1.2")
    SAR_DIR = os.path.join(BASE_DIR, "data", "flood_events", "HandLabeled", "S1Hand")
    MASK_DIR = os.path.join(BASE_DIR, "data", "flood_events", "HandLabeled", "LabelHand")
    
    train_csv_path = os.path.join(BASE_DIR, "splits", "flood_handlabeled", "flood_train_data.csv")
    test_csv_path = os.path.join(BASE_DIR, "splits", "flood_handlabeled", "flood_test_data.csv")
    
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        df_train = pd.read_csv(train_csv_path)
        train_chip_ids = df_train[df_train.columns[0]].tolist()
        print(f"Successfully loaded {len(train_chip_ids)} training chips.")
        
        df_test = pd.read_csv(test_csv_path)
        test_chip_ids = df_test[df_test.columns[0]].tolist()
        print(f"Successfully loaded {len(test_chip_ids)} testing chips.")
        
        trained_model = train_pipeline(SAR_DIR, MASK_DIR, train_chip_ids, device)
        test_and_visualize_pipeline(trained_model, SAR_DIR, MASK_DIR, test_chip_ids, device)
    else:
        print(f"\nERROR: Could not find CSV splits.")