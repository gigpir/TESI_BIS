# **Lista dei file:**

######I file `.pkl` possono essere aperti e chiusi utilizzando i wrapper `load_data` e `save_data` che si trovano nel file `primary/data_io.py` 

### Dataset con coordinate t-SNE:
- #### artist_m{i}_hm.pkl
  **TIPO:** Dizionario di oggetti `Artist` _(fare riferimento alla classe descritta nel file `primary/data_class.py`)_.\
  Ad ogni oggetto della classe `Artist` è _opzionalmente_ associata una heatmap 20x20 accessibile tramite la proprietà `.tsne_heatmap`.\
  Ad ogni oggetto della classe `Artist` è associato un dizionario di oggetti `Song` la cui chiave è l'id del brano.\
  Ad ogni oggetto della classe `Song` è _opzionalmente_ associata una tupla contenente le coordinate tsne accessibile tramite la proprietà `.tsne[0]` o `.tsne[1]`\
  Al variare della modalità m varia il modo in qui è stato calcolato t-SNE
  - `m0` media
  - `m1` media + varianza
  - `m2` media + varianza + derivate prime
  - `m3` media + varianza + derivate prime + derivate seconde
  
### File derivati da artist_m3_hm.pkl:
- #### distances_cc_peak_1.pkl 
    **TIPO:** Pandas DataFrame.\
    Contiene le distanze calcolate con il criterio
    che fa uso della cross correlazione considerando un solo picco.\
    La funzione implementa la metrica è `compute_cross_correlation_distance`\
    Per accedere alla distanza tra l'artista 'a' e l'artista 'b' digitare l'istruzione: `df['a']['b']` dopo aver caricato il file. 

- #### distances_cc_peak_2.pkl 
    **TIPO:** Pandas DataFrame.\
    Contiene le distanze calcolate con il criterio
    che fa uso della cross correlazione considerando un solo picco e dividendo poi il valore dello shift per il picco stesso.\
    La funzione implementa la metrica è `compute_cross_correlation_distance_normalized`\
    Per accedere alla distanza tra l'artista 'a' e l'artista 'b' digitare l'istruzione: `df['a']['b']` dopo aver caricato il file.

- #### max_length_ranking_cc_peak_1.pkl e max_length_ranking_cc_peak_2.pkl
  **TIPO:** dizionario di ranking.\
  **Chiave**: id artista.\
  **Valore**: ranking, una lista ordinata di id artista \
- #### ground_truth.pkl 
  **TIPO:** dizionario di ranking.\
  **Chiave**: id artista.\
  **Valore**: ranking, una lista ordinata di id artista \
   
- #### heatmaps.pkl
  **TIPO:** dizionario di matrici numpy.\
  **Chiave**: id artista.\
  **Valore**: opzionale, heatmap, matrice numpy 20x20. Gli artisti che non possiedono una heatmap hanno esplicitamente il valore `None` (null)
        
- ####names.pkl
  **TIPO:** dizionario di stringhe.\
  **Chiave**: id artista.\
  **Valore**: stringa corrispondente al nome dell'artista
  

# **Script**

###### Il progetto che contiene gli script `.py` elencati qui sotto si trova nella directory `/home/crottondi/PIRISI_TESI/`.
###### Gli script `.sbatch` necessari alla sottomissione dei job si trovano nella directory `/home/crottondi/4_SBATCH_COMPACT`


### Generazione matrice delle distanze:
  Per svolgere questa operazione sul dataset completo è necessario suddividere il lavoro su più nodi. Purtroppo non sono riuscito ad integrare i moduli MPI e singularity sul cluster
  quindi l'implementazione 'multinodo' è statica e si articola in queste fasi.
- #### Generazione chunk
  Viene usato `3_RANKING/1_generate_N_chunks.py` per generare `n` chunk del tipo `chunk_<i>.pkl`
  in una cartella output.\
  Ogni file `chunk_<i>.pkl` contiene una lista di id artista. Una volta che si decide il numero di chunk non è necessario
  avviare ogni volta questo script.
  ###### Script `.sbatch`: fare riferimento a `/home/crottondi/PIRISI_TESI/4_SBATCH_COMPACT/4_COMPACT_create_chunks.sbatch`
  
- #### Calcolo delle distanze 
  `4_SCRIPTS/build_distances_chunk.py` è uno script che lavora in multiprocessing (livello nodo). Riceve come parametri 
  la metrica, il dataset, e il chunk assegnato.\
  Produce in output un file `.pkl` contenente un dizionario di dizionari.\
  Laddove si voglia cambiare la metrica di confronto tra le heatmap occorre aggiungere un valore per parametro 
  metric nel main e modificare opportunamente la funzione `build_matrix_slave`.    
  ###### Script `.sbatch`: fare riferimento a `/home/crottondi/PIRISI_TESI/4_SBATCH_COMPACT/4_COMPACT_build_distances_cc_peak_<1>_master.sh`  
- #### Merge chunk
  `4_SCRIPTS/merge_chunks.py` riceve come parametri la cartella contente i `chunk_<i>_OUT.pkl` e il numero di chunk.\
  Produce in output un file `.pkl` contenente un DataFrame Pandas. 
  I file `distances_cc_peak_1.pkl ` e `distances_cc_peak_2.pkl` ne sono un esempio.
  ###### Script `.sbatch`: fare riferimento a `/home/crottondi/PIRISI_TESI/4_SBATCH_COMPACT/4_COMPACT_merge_chunks.sbatch`

### Generazione `ground_truth.pkl`, `heatmaps.pkl`, `names.pkl` :
  ###### Script `.py`: `4_SCRIPTS/build_names_heatmap_gt.py`
  ###### Script `.sbatch`: fare riferimento a `/home/crottondi/PIRISI_TESI/4_SBATCH_COMPACT/4_COMPACT_build_names_heatmap_gt.sbatch`

### Generazione `max_length_ranking_cc_peak_1.pkl` e `max_length_ranking_cc_peak_2.pkl`:
  ###### Script `.py`: `4_SCRIPTS/build_names_heatmap_gt.py`
  ###### Script `.sbatch`: fare riferimento a `/home/crottondi/PIRISI_TESI/4_SBATCH_COMPACT/4_COMPACT_build_names_heatmap_gt.sbatch`