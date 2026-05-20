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
        
        # Determinam coloana de imagini (pentru alunite e 'imagine', pentru fete e 'path' sau 'id')
        if 'imagine' in self.data_frame.columns:
            self.img_col = 'imagine'
        elif 'path' in self.data_frame.columns:
            self.img_col = 'path'
        elif 'id' in self.data_frame.columns:
            self.img_col = 'id'
        else:
            raise KeyError("Nu s-a putut determina coloana cu caile/numele imaginilor.")
            
        self.img_names = self.data_frame[self.img_col].astype(str).tolist()

        
        if not self.is_test:
            self.label_col = 'diagnostic' if 'diagnostic' in self.data_frame.columns else 'label'
            self.labels = self.data_frame[self.label_col].tolist()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Numele sau calea relativa a imaginii
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Incarcam si fortam sa fie RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            # Pentru submiterile finale Kaggle
            return image, img_name
            
        # Extragem eticheta
        label = self.labels[idx]
        
        # Daca avem un mapping explicit (cum e la alunite unde avem string-uri)
        if self.label_mapping is not None:
            label = self.label_mapping[label]
        else:
            # Fete: etichetele sunt 1-7 (numpy.int64 sau str) -> facem 0-indexate
            label = int(label) - 1

        return image, label

    def get_labels(self):
        """Returneaza toate etichetele din dataset fara a incarca imaginile.
        Util pentru calculul class weights / sampler weights.
        """
        if self.is_test:
            return []
        labels = []
        for label in self.labels:
            if self.label_mapping is not None:
                label = self.label_mapping[label]
            else:
                label = int(label) - 1
            labels.append(int(label))
        return labels

