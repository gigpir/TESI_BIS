## **Lista dei file:**

##### tutti i file .pkl contengono dei dizionari la cui chiave è sempre l'id dell'artista

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