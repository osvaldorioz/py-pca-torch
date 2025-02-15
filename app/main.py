from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import pca_extension
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/principal-component-analysis")
def calculo(samples: int, characteristics: int):
    output_file = 'principal-component-analysis.png'
    # Generar datos de ejemplo
    torch.manual_seed(0)
    data = torch.randn(samples, characteristics)  # x muestras, y características

    # Aplicar PCA para reducir a 2 componentes principales
    num_components = 2
    transformed_data = pca_extension.pca(data, num_components).numpy()

    # Graficar los datos transformados
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='blue', label='Datos proyectados')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA: Proyección a 2 Componentes Principales')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/principal-component-analysis-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)