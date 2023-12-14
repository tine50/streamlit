# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from my_functions.functions_cached import *  # personnal functions pkg and module
import streamlit.components.v1 as components

#######################################################################################
# To run this code, type in terminal at the file path: 
# streamlit run credit-scoring.py


#######################################################################################
# Setting layout & navigation pane
st.set_page_config(page_title="Cat Credit Score Dashboard", # Must be 1st st statement
                    page_icon="☮",
                    initial_sidebar_state="expanded")

# defining containers of the app
header = st.container() 
dataset = st.container()
eda = st.container()
model_training = st.container()
model_predict = st.container()
# Connexion page 
def connexion():
    with header:
        # HTML et CSS pour la flèche
        flèche_html = """
        <style>
        #flèche-haut {
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            background-color: #007bff;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 40px;
            font-size: 20px;
            cursor: pointer;
            z-index: 9999;
        }
        </style>

        <div id="flèche-haut">
            &#8679; 
        </div>
        <p>User : mister_tine, Password : 123</p>
        """
        

        # JavaScript pour le défilement vers le haut
        script_js = """
        <script>
        document.getElementById("flèche-haut").addEventListener("click", function() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        });
        </script>
        """

        components.html(script_js, height=0)
        st.markdown(flèche_html, unsafe_allow_html=True)
        st.title("Connexion")
        
        # Charger l'image
        image = Image.open('assets/banner.jpg')

        # Redimensionner l'image
        image = image.resize((1200, 300))

        # Afficher l'image redimensionnée
        st.image(image)
        # st.image("assets/banner.jpg", width=100)
        st.header("** Abdoussalam TINE **")
        st.markdown("L'objectif de ce projet :")
        st.markdown(" Mise en place d'un outil de scoring de crédit destiné à évaluer la probabilité de remboursement d'un client, accompagné d'un tableau de bord interactif. Ceci permet de classer les demandes de crédit en tant qu'accordées ou refusées.")
    #st.markdown("Let's start")
    # Créer des champs pour les identifiants
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type='password')

    # Bouton de connexion
    if st.sidebar.button("Connexion"):
        # Vérification des identifiants (remplacer par votre propre logique)
        if username == "mister_tine" and password == "123":
            st.success("Connecté en tant que {}".format(username))
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.session_state['authenticated'] = False
            st.error("Identifiants incorrects")

    
    
    
def main():
    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        show_main_content()
    else :
        connexion()
        
def show_main_content():
#######################################################################################
    # Stating graphical parameters
    COLOR_BR_r = ['#00CC96', '#EF553B'] #['dodgerblue', 'indianred']
    COLOR_BR = ['indianred', 'dodgerblue']
    #######################################################################################

    # Managing data import/export
    PATH = os.getcwd() #os.getcwd()+'/' # change for online app or local
    PATH_INPUT = PATH+"/input/"
    FILENAME_TRAIN = PATH_INPUT+'application_train.csv' # sample of train set for online version 25MB
    FILENAME_TEST = PATH_INPUT+'application_test.csv'
    FILENAME_MODEL = PATH_INPUT+'XGBoost_with_threshold.pkl' # PATH

    df_train = get_data(FILENAME_TRAIN) # load trainset data in a df
    df_test = get_data(FILENAME_TEST) # load testset (unlabeled) data in a df

    # Logout button
    if st.sidebar.button("Déconnexion"):
        st.session_state['authenticated'] = False
        st.experimental_rerun()  # Rerun the script to update the page


    sb = st.sidebar # add a side bar 
    sb.image('assets/cat.jpeg', width=300)
    rad_who = sb.radio('', ['👨🏽‍💼 Salarié']) 
    # the two versions of the app will have different options, home is common to all
    if rad_who == '👨🏽‍💼 Salarié': 
        np.random.seed(13) # one major change is that client is directly asked as input since sidebar
        label_test = df_test['SK_ID_CURR'].sample(50).sort_values()
        radio = sb.radio('', ['Listes des ID Client']) #'Renseigner le Client'
        if radio == 'Listes des ID Client': # Choice choose preselected seed13 or a known client ID
            input_client = sb.selectbox('Listes des ID Client', label_test)
        #if radio == 'Renseigner le Client':
            #input_client = int(sb.text_input('Renseigner le Client', value=147254))

        sb.markdown('**Navigation**')
        rad = sb.radio('', ['🏠 Accueil', 
        '🔎 Données Client',
        '👁️ Une Vue Global', 
        '🚀 Prediction Client'])
    else:
        sb.markdown('**Navigation**')
        rad = sb.radio('', ['🏠 Accueil'])

    # defining containers of the app
    header = st.container() 
    dataset = st.container()
    eda = st.container()
    model_training = st.container()
    model_predict = st.container()

    #######################################################################################
    # Implementing containers
    #######################################################################################

    if rad == '🏠 Accueil': # with this we choose which container to display on the screen
        with header:

            
            st.header("** Abdoussalam TINE **")
            st.markdown("L'objectif de ce projet :")
            st.markdown(" Mise en place d'un outil de scoring de crédit destiné à évaluer la probabilité de remboursement d'un client, accompagné d'un tableau de bord interactif. Ceci permet de classer les demandes de crédit en tant qu'accordées ou refusées.")
            #st.markdown("Let's start")
            st.title("Cat Credit Score Dashboard! \n ----")
            # Charger l'image
            image = Image.open('assets/banner.jpg')

            # Redimensionner l'image
            image = image.resize((1200, 300))

            # Afficher l'image redimensionnée
            st.image(image)
            st.markdown("- 👨🏽‍💼 Salarié")


    #######################################################################################

    if rad == '👁️ Une Vue Global':
        with dataset:
            st.header("**Analyse Courte.** \n ----") # title > header > subheader > markdown ~ text
            #st.markdown("In this project, we focus only on the application train dataset.")
            
            st.subheader("Données Global.")
            max_row = st.slider("Merci de selectionner le nombre de client que vous souhaitez afficher", value=1000, min_value=1,
                                max_value=len(df_train)) 
            st.write(df_train.head(max_row))
            

            val_notnull = df_train.count().sum()
            lig,col = df_train.shape
            taux = val_notnull/(lig*col)*100  
            # Informations générales pour le diagramme circulaire 
            fig, ax = plt.subplots(figsize=(5, 5))
            st.title("Taux de remplissage") 
            ax.axis("equal") 
            # Ploter la figure pie
            ax.pie([taux, 100 - taux], labels=["Taux de remplissage", "Taux de valeurs manquantes"],autopct='%1.2f%%',explode=(0,0.1),radius=1)
            #st.write(["Taux de remplissage", "Taux de valeurs manquantes"])
            st.pyplot(fig)
            
    ##########################################################################################################################
    if rad == '🔎 Données Client': 
        with eda:
            st.header("**Données Client.** \n ----")
            # retrieving whole row of client from sidebar input ID
            client_data = df_test[df_test.SK_ID_CURR == input_client]
            client_data = client_data.dropna(axis=1) # avoiding bugs

            st.subheader(f"**Client ID: {input_client}.**")
            # plotting features from train set, with client's data as dashed line (client!=None in func)
            st.subheader("Ranking client in some features.")      
            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(histogram(df_train, x='CODE_GENDER', client=[df_test, input_client]), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x='EXT_SOURCE_1', client=[df_test, input_client]), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_2', client=[df_test, input_client]), use_container_width=True)

            st.subheader("Souhaitez-vous realisez quelques analyses?.")
            col1, col2, col3 = st.columns(3)
            num_col = client_data.select_dtypes(include=np.number).columns.sort_values()
            input1 = col1.selectbox('1st plot', num_col)
            input2 = col2.selectbox('2nd plot', num_col[1:])
            input3 = col3.selectbox('3rd plot', num_col[2:])

            st.subheader("Vous pouvez realisez davantage d'analyses bivariées.")
            col4, col5, col6 = st.columns(3)
            cat_col = client_data.select_dtypes(exclude=np.number).columns.sort_values()
            input4 = col4.selectbox('1st plot', cat_col[1:])
            input5 = col5.selectbox('2nd plot', cat_col[2:])
            input6 = col6.selectbox('3rd plot', cat_col[3:])

            button = st.button('Plot it! ')
            if button:
                col1.plotly_chart(histogram(df_train, x=input1, legend=False, client=[df_test, input_client]),use_container_width=True)
                col2.plotly_chart(histogram(df_train, x=input2, legend=False, client=[df_test, input_client]),use_container_width=True)
                col3.plotly_chart(histogram(df_train, x=input3, legend=False, client=[df_test, input_client]),use_container_width=True)
                col4.plotly_chart(histogram(df_train, x=input4, legend=False, client=[df_test, input_client]),use_container_width=True)
                col5.plotly_chart(histogram(df_train, x=input5, legend=False, client=[df_test, input_client]),use_container_width=True)
                col6.plotly_chart(histogram(df_train, x=input6, legend=False, client=[df_test, input_client]),use_container_width=True)
            
            st.subheader("Plus d'information sur ce client.")
            # displaying values from a dropdown (had issues with NaNs that's why .dropna())
            col1, col2 = st.columns(2)
            info = col1.selectbox('Menons notre enquete sur ce client ?', client_data.columns.sort_values())     
            info_print = client_data[info].to_numpy()[0]

            col1.subheader(info_print)
            # displaying whole non NaNs row
            col2.write("Une vue à 360° sur ce client.")
            col2.write(client_data)

    ###########################################################################################################################
    if rad == '🚀 Prediction Client': 
        with model_predict:
            st.header("**La Capacité du client à etre solvable.** \n ----")

            col1, col2 = st.columns(2)
            col1.markdown(f'**Client ID: {input_client}**')

            if col2.button('Prediction & plot!'):
                # this time we need all outputs of preprocessing                    
                X_train_sc, X_test_sc, feat_list = preprocess(df_train, df_test)
                y_train = df_train['TARGET']
                # calling pretrained model from pickle file (.sav)
                try: 
                    model = pickle.load(open(FILENAME_MODEL, 'rb'))
                except:
                    raise 'You must train the model first.'
                # finding client row index in testset
                idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_client].index
                client = X_test_sc[idx, :] # for then slicing preprocessed test data
                
                y_prob = model.predict_proba(client) # predicting proba
                y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]] # misalignement of format
                # importance of features extracted using scikit learn: pred_contrib=True
                imp_feat = model.predict_proba(X_test_sc[idx, :], pred_contrib=True).flatten()
                imp = pd.DataFrame([feat_list, imp_feat]).T.sort_values(by=1, ascending=False).head(20)

                col1, col2 = st.columns(2)
                # adapting message wether client's pos or neg
                if y_prob[1] < y_prob[0]:
                    col1.subheader(f"**Pret Accordé!!.**")
                    st.balloons()
                    st.success(f"**Ce client à une forte capacité de remboursement !!.")
                    #st.text(f"**Ce client à une forte capacité de remboursement !!.")
                else:
                    col1.subheader(f"**Desolé, Pret non accordé.**")
                    #st.text(f"**Ce client présente un déficit de payement trés impportant !!.")
                    st.error(f"**Ce client présente un déficit de payement trés important !!.")
                # plotting pie plot for proba, finding good h x w was a bit tough
                fig = px.pie(values=y_prob, names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r, 
                width=230, height=230)
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                col1.plotly_chart(fig, use_container_width=True)

                col2.subheader("**Client spiderchart.**")
                # plotting radar chart
                columns = (imp.head(5)[0].values) # recovering top5 most important features as... tuples, why did I do that???
                df_test_sc = pd.DataFrame(X_test_sc, columns=feat_list)
                # I wanted to plot average that's why I made a df, but I think it's useless now 
                # since it was a bit difficult and I drop this idea. Instead, I kept scaled version of data
                # so average should be zero and 1 = +1 sigma (StandardScaler)
                client_radar = df_test_sc.loc[idx,columns].T.reset_index()
                client_radar = client_radar.rename(columns={"index":"theta", idx.values[0] :'r'})

                fig = px.line_polar(client_radar, 
                                    theta='theta', 
                                    r='r', 
                                    log_r=False, 
                                    line_close=True,
                                    color_discrete_sequence=['indianred'],
                                    width=250,
                                    height=250,
                                    )
                fig.update_traces(fill='toself')
                fig.update_layout(margin=dict(l=50, r=50, t=50, b=10))  
                col2.plotly_chart(fig, use_container_width=True)

                st.subheader("**Les variables qui nous confortent sur notre decision.**")
                # then plotting feature importance, but for readibility slicing absissa labels using:
                labels = [(i[:7] + '...'+i[-7:]) if len(i) > 17 else i for i in imp[0]]
                fig = px.bar(   imp.head(10),
                                x=0,
                                y=1,
                                width=300,
                                height=300,
                                color=range(10),
                                color_continuous_scale='OrRd_r',
                                orientation='v')
                fig.update(layout_coloraxis_showscale=False)
                fig.update_xaxes(title='')
                fig.update_layout(xaxis = dict(
                                tickmode = 'array',
                                tickvals = [i for i in range(20)],
                                ticktext = labels))
                fig.update_yaxes(title='Relative importance')
                fig.update_yaxes(showticklabels=False)
                fig.update_layout(margin=dict(l=20, r=20, t=10, b=10))                
                st.plotly_chart(fig, use_container_width=True)

                # one-hot-encoded columns added a "_string" as lower case to col names
                # thus checking if the col name is full upper case if a good test to 
                # check whether the col is num or cat (I want only 6 most num feats here)
                num_plots=[]
                i=0
                num_plots = pd.Series(dtype=object)  # Initialize an empty Pandas Series
                while (i in range(len(imp))) and (len(num_plots) < 7):
                    if imp.iloc[i, 0] == imp.iloc[i, 0].upper():
                        # Create a new Series for the current item and concatenate it
                        item_series = pd.Series([imp.iloc[i, 0]])
                        num_plots = pd.concat([num_plots, item_series]).reset_index(drop=True)
                    i += 1
                    
# Script JavaScript pour fermer la barre latérale sur les écrans mobiles
script = """
    <script>
    const mediaQuery = window.matchMedia('(max-width: 768px)');
    if (mediaQuery.matches) {
        window.onload = () => {
            const sidebar = document.querySelector('.eyeqlp51.st-emotion-cache-fblp2m.ex0cdmw0');
            if (sidebar) {
                sidebar.click();
            }
        }
    }
    </script>
"""


# Injecter le script dans la page
st.markdown(script, unsafe_allow_html=True)

# components.html(icon, height=0)
##########################################################################################################################
if __name__ == "__main__":
    # if 'authenticated' not in st.session_state:
    #     st.session_state['authenticated'] = False
    main()
