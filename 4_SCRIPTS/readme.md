

# **Lista dei file:**

###### Tutti i file _.pkl_ contengono un dizionario la cui chiave è l'id dell'artista

### Dataset t-SNE:
- #### artist_m{i}_hm.pkl
  - contiene un dizionario di oggetti `Artist` (fare riferimento alla classe descritta nel file `primary/data_io.py`).
  - Ad ogni oggetto della classe `Artist` è _opzionalmente_ associata una heatmap 20x20 accessibile tramite la proprietà `.tsne_heatmap`.
  - Ad ogni oggetto della classe `Artist` è associato un dizionario di oggetti `Song` la cui chiave è l'id del brano.
  - Ad ogni oggetto della classe `Song` è _opzionalmente_ associata una tupla contenente le coordinate tsne accesssibile tramite la proprietà `.tsne[0]` o `.tsne[1]`  
- #### distances_cc_peak_1.pkl 
    dizionario di dizionari (nested) che contiene le distanze calcolate con il criterio
    che fa uso della cross correlazione considerando un picco

- #### distances_cc_peak_2.pkl 
    dizionario di dizionari (nested) che contiene le distanze calcolate con il criterio
    che fa uso della cross correlazione considerando un picco normalizzando 
    per il valore del picco
  
        
**esempio**: una volta caricato il file è possibile accedere alla distanza tra artista id_1 e id_2 con la seguente sintassi:
    
    dist = dizionario['id_1']['id_2']    

- #### max_length_ranking_cc_peak_1.pkl e max_length_ranking_cc_peak_2.pkl
    dizionario di ranking, ogni ranking è una lista ordinata di id artista 
- #### ground_truth.pkl 
    contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una lista di id
   
- #### heatmaps.pkl 
    contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una matrice numpy 20x20
    gli artisti che non possiedono una heatmap hanno esplicitamente il valore None (null)
        
- ####names.pkl
    contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una stringa col nome dell'artista