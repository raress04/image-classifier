import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CSVImageDataset(Dataset):
    """
    Dataset generic pentru incarcarea imaginilor si etichetelor dintr-un fisier CSV.
    Acomodeaza atat setul de date cu alunite, cat si pe cel cu fete.
    """
    def __init__(self, csv_file, img_dir, transform=None, is_test=False, label_mapping=None):
        """
        Args:
            csv_file (str): Calea catre fisierul CSV cu adnotari.
            img_dir (str): Directorul care contine imaginile (sau baza de unde incep caile din CSV).
            transform (callable, optional): Transformari optionale (PyTorch transforms).
            is_test (bool): True daca este setul de test de pe Kaggle (care nu are etichete).
            label_mapping (dict, optional): Un dictionar pentru a mapa stringurile (ex. 'nv') in intregi.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.label_mapping = label_mapping
        
        # Determinam coloana de imagini (pentru alunite e 'imagine', pentru fete e 'path')
        self.img_col = 'imagine' if 'imagine' in self.data_frame.columns else 'path'
        
        if not self.is_test:
            self.label_col = 'diagnostic' if 'diagnostic' in self.data_frame.columns else 'label'

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Numele sau calea relativa a imaginii
        img_name = str(self.data_frame.iloc[idx][self.img_col])
        img_path = os.path.join(self.img_dir, img_name)
        
        # Incarcam si fortam sa fie RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            # Pentru submiterile finale Kaggle
            return image, img_name
            
        # Extragem eticheta
        label = self.data_frame.iloc[idx][self.label_col]
        
        # Daca avem un mapping explicit (cum e la alunite unde avem string-uri)
        if self.label_mapping is not None and label in self.label_mapping:
            label = self.label_mapping[label]
        elif isinstance(label, str) and label.isdigit():
            # Pentru dataset-ul de fete etichetele sunt '1', '2' etc. (le facem 0-indexate)
            label = int(label) - 1

        return image, label
