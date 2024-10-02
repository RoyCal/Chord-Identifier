import queue
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import scipy as sp
import matplotlib.pyplot as plt

import librosa
import librosa.display as ld
from IPython.display import Audio

import csv
import itertools
from collections import Counter

def maiores_indices(lista, n):
    if n > len(lista):
        raise ValueError("O número de índices solicitados excede o tamanho da lista.")
    
    # Criar uma lista de tuplas (índice, valor)
    lista_com_indices = list(enumerate(lista))
    
    # Ordenar a lista pelo valor, em ordem decrescente
    lista_ordenada = sorted(lista_com_indices, key=lambda x: x[1], reverse=True)
    
    # Retornar os primeiros 'n' índices
    return [indice for indice, valor in lista_ordenada[:n]]

def carregar_frequencias(arquivo_csv):
    notas = []
    frequencias = []
    
    with open(arquivo_csv, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        notas = next(reader)
        for linha in reader:
            frequencias.append([float(freq) for freq in linha])
    
    return notas, frequencias

notas, frequencias = carregar_frequencias('Tabelas/frequencias.csv')

def carregar_formacoes(arquivo_csv):
    tipos_de_acorde = []
    formacoes = []
    
    with open(arquivo_csv, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        tipos_de_acorde = next(reader)
        for linha in reader:
            formacoes.append([freq for freq in linha])
    
    return tipos_de_acorde, formacoes

tipos_de_acorde, formacoes = carregar_formacoes("Tabelas/formacoes.csv")

def get_chord(formation):
    global tipos_de_acorde
    global formacoes

    for linha in formacoes:
        for i, notes in enumerate(linha):
            if formation in notes:
                if i == 0:
                    return "Nota: " + linha[0]
                elif i == 1:
                    return linha[0]
                else:
                    return linha[0] + tipos_de_acorde[i]
    
    return None

def nota_mais_proxima(frequencia):
    global notas
    global frequencias
    
    diferenca_minima = float('inf')
    nota_proxima = None
    
    for i, faixa in enumerate(frequencias):
        for j, freq in enumerate(faixa):
            diferenca = abs(frequencia - freq)
            if diferenca < diferenca_minima:
                diferenca_minima = diferenca
                nota_proxima = notas[j]

    return nota_proxima

def get_formation(note_list):
    remove_duplicates = set(note_list)
    formation = "".join(sorted(list(remove_duplicates)))
    return (formation, len(remove_duplicates))

def freqs_to_notes(freq_list):
    notes = []
    for freq in freq_list:
        notes.append(nota_mais_proxima(freq))
    
    return notes

def gerar_combinacoes(lista, n):
    lista = sorted(lista)

    combinacoes = list(itertools.combinations(lista, n))

    formacoes_possiveis = []
    for tupla in combinacoes:
        formacoes_possiveis.append("".join(tupla))
    
    return formacoes_possiveis

device = 0
window = 1000
downsample = 1
channels = [1]
interval = 30

q = queue.Queue()

device_info = sd.query_devices(device, "input")
samplerate = device_info["default_samplerate"]
length = int(window*samplerate/(1000*downsample))

plotdata = np.zeros((length, len(channels)))
data_dct = np.zeros(length)

fig, ax = plt.subplots(figsize=(16, 8))

lines = ax.plot(plotdata, color = (0, 1, 0.29))

def audio_callback(indata, frames, time, status):
    q.put(indata[::downsample, [0]])

def update_plot(frame):
    global plotdata
    global data_dct

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

        new_data = data.squeeze()

        data_dct = np.roll(data_dct, -shift)
        data_dct[-shift:] = new_data
        f = abs(sp.fft.dct(data_dct))
        lista = maiores_indices(f,10)
        lista = freqs_to_notes(lista)
        formation, n_notes = get_formation(lista)
        chord = get_chord(formation)

        if chord is not None:
            print(chord)
        elif n_notes == 3:
            print("Não foi possível identificar um acorde com essa formação")
        elif n_notes == 4:
            formacoes_possiveis = gerar_combinacoes(set(lista), 3)

            for formacao in formacoes_possiveis:
                chord = get_chord(formacao)
                if chord != None:
                    print(chord)
        elif n_notes > 4:
            formacoes_possiveis = gerar_combinacoes(set(lista), 4) + gerar_combinacoes(set(lista), 3)

            for formacao in formacoes_possiveis:
                chord = get_chord(formacao)
                if chord != None:
                    print(chord)
        elif n_notes == 2:
            print(get_chord(Counter(lista).most_common(1)[0][0]))

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

ax.set_facecolor((0, 0, 0))
ax.set_yticks([0])
ax.yaxis.grid(True)

stream = sd.InputStream(device=device, channels=max(channels), samplerate=samplerate, callback=audio_callback)

ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)

with stream:
    plt.show()