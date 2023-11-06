#--------------------LIBRERÍAS----------------------------#
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pycaret.classification import load_model, predict_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from plotly.subplots import make_subplots



#------------------------------------------------#


st.set_page_config(page_title='League of Legends', page_icon=':space_invader:', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

#--------------------Imagenes----------------------------#
image_title = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\LoL_Logo_Rendered_LARGE.png')
image_riot_logo = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\riot_logo.png')
image_lol_logo = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\eu2-map.jpg')
image_region = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\eu2-map.jpg')
image_play = 'https://brand.riotgames.com/static/5d967107520142f4f9cf8798900614b4/ed70a/04_MagicFundamentals_2.webp'
image_map = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\Summoner%27s_Rift_Minimap.webp')
image_map2 = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\Minimap.jpg')
legal = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\legal.png')
players_league = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\other\players_league.png')

# Graph
# average_players = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\Number-of-monthly-LoL-players-per-years.png.webp')

# Champs
champ_riven = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\champions\riven.png')
champ_ryze = Image.open(r'C:\Users\Javi\Desktop\Cloned_repo\League-of-Legends-ML\img\champions\ryze.png')
# Leagues

image_iron = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\iron.png')
image_bronze = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\bronze.png')
image_silver = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\silver.png')
image_gold = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\gold.png')
image_plat = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\platinum.png')
image_dia = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\diamond.png')
image_master = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\master.png')
image_grandmaster = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\grandmaster.png')
image_challenger = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\leagues\challenger.png')

# Lanes
image_top = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\lanes\icon-position-top.png')
image_jungle = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\lanes\lol_jungle_icon_by_divoras_degndao.png')
image_mid = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\lanes\Middle_icon.png')
image_adc = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\lanes\icon-position-bottom.png')
image_support = Image.open(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\lanes\icon-position-utility.png')

#--------------------Videos----------------------------#

DEFAULT_WIDTH = 80
VIDEO_DATA1 = r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\champions\Senna_Dance012.mp4'
VIDEO_DATA2 = 'https://assets.contentstack.io/v3/assets/blt2ac872571a60ee02/blt800704d25197b0ce/61787ba52a814718581ad02f/Yasuo_Idle2.mp4'
VIDEO_DATA3 = 'https://assets.contentstack.io/v3/assets/blt2ac872571a60ee02/blt583cc47a487e55ab/617746705422ab67be73e683/Poro_base_AN_idle3.mp4'
VIDEO_DATA4 = 'https://assets.contentstack.io/v3/assets/blt2ac872571a60ee02/blt8f30adcf1a5fae2d/61787b31fdb9af22b36e3aa8/Yasuo_RunHaste.mp4'
VIDEO_DATA5 = r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\img\league_of_legends_logo_animation (720p).mp4'

#--------------------Links----------------------------#
link = '<iframe title="dashboard_airbnb_tokyo" width="1140" height="541.25" src="https://app.fabric.microsoft.com/reportEmbed?reportId=9396b597-3eba-4262-bc18-295004ae53e6&autoAuth=true&ctid=8aebddb6-3418-43a1-a255-b964186ecc64" frameborder="0" allowFullScreen="true"></iframe>'

#--------------------DATA----------------------------#
players = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\players\players.csv')
matches = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches.csv')
players_clean = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\players\players_clean')
matches_clean = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_clean')
winrate_league = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\players\winrate_league')

# Duration
iron_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\iron_matches_duration')
bronze_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\bronze_matches_duration')
silver_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\silver_matches_duration')
gold_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\gold_matches_duration')
plat_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\plat_matches_duration')
dia_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\dia_matches_duration')
master_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\master_matches_duration')
grandmaster_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\grandmaster_matches_duration')
challenger_matches_duration = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\matches_duration\challenger_matches_duration')

# Winrate
iron_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\iron\iron_matches_top_champions_winratio')
iron_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\iron\iron_matches_jungle_champions_winratio')
iron_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\iron\iron_matches_mid_champions_winratio')
iron_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\iron\iron_matches_adc_champions_winratio')
iron_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\iron\iron_matches_support_champions_winratio')

bronze_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\bronze\bronze_matches_top_champions_winratio')
bronze_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\bronze\bronze_matches_jungle_champions_winratio')
bronze_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\bronze\bronze_matches_mid_champions_winratio')
bronze_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\bronze\bronze_matches_adc_champions_winratio')
bronze_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\bronze\bronze_matches_support_champions_winratio')

silver_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\silver\silver_matches_top_champions_winratio')
silver_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\silver\silver_matches_jungle_champions_winratio')
silver_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\silver\silver_matches_mid_champions_winratio')
silver_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\silver\silver_matches_adc_champions_winratio')
silver_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\silver\silver_matches_support_champions_winratio')

gold_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\gold\gold_matches_top_champions_winratio')
gold_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\gold\gold_matches_jungle_champions_winratio')
gold_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\gold\gold_matches_mid_champions_winratio')
gold_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\gold\gold_matches_adc_champions_winratio')
gold_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\gold\gold_matches_support_champions_winratio')

plat_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\plat\plat_matches_top_champions_winratio')
plat_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\plat\plat_matches_jungle_champions_winratio')
plat_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\plat\plat_matches_mid_champions_winratio')
plat_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\plat\plat_matches_adc_champions_winratio')
plat_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\plat\plat_matches_support_champions_winratio')

dia_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\dia\dia_matches_top_champions_winratio')
dia_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\dia\dia_matches_jungle_champions_winratio')
dia_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\dia\dia_matches_mid_champions_winratio')
dia_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\dia\dia_matches_adc_champions_winratio')
dia_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\dia\dia_matches_support_champions_winratio')

master_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\master\master_matches_top_champions_winratio')
master_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\master\master_matches_jungle_champions_winratio')
master_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\master\master_matches_mid_champions_winratio')
master_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\master\master_matches_adc_champions_winratio')
master_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\master\master_matches_support_champions_winratio')

grandmaster_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\grandmaster\grandmaster_matches_top_champions_winratio')
grandmaster_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\grandmaster\grandmaster_matches_jungle_champions_winratio')
grandmaster_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\grandmaster\grandmaster_matches_mid_champions_winratio')
grandmaster_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\grandmaster\grandmaster_matches_adc_champions_winratio')
grandmaster_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\grandmaster\grandmaster_matches_support_champions_winratio')

challenger_matches_top_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\challenger\challenger_matches_top_champions_winratio')
challenger_matches_jungle_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\challenger\challenger_matches_jungle_champions_winratio')
challenger_matches_mid_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\challenger\challenger_matches_mid_champions_winratio')
challenger_matches_adc_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\challenger\challenger_matches_adc_champions_winratio')
challenger_matches_support_champions_winratio = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\matches\champions_winrate\challenger\challenger_matches_support_champions_winratio')
#--------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
st.image(image_title, caption='League of legends logo')

#--------------------SIDEBAR----------------------------#

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.image(image_riot_logo, width=100)

with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.image(image_lol_logo, width=100)


st.sidebar.markdown('-------------------------------------------------------------------------------------', unsafe_allow_html=True)


st.sidebar.markdown("<h1 style='text-align: left; font-size: 25px; font-family: Beaufort;'>Table of Contents</h1>", unsafe_allow_html=True)

# Buttons concept
# button_intro = st.sidebar.button('Introduction')
# button_collect = st.sidebar.button('Data collection')
# button_preprocessing = st.sidebar.button('Preprocessing')
# button_eda = st.sidebar.button('Exploratory Data Analysis')
# button_ml = st.sidebar.button('Machine Learning')
# button_dashboard = st.sidebar.button('Dashboard')
# button_conclusions = st.sidebar.button('Conclusions')

selection = st.sidebar.radio('', ['Introduction','Data collection', 'Preprocessing', 'Exploratory Data Analysis', 'Machine Learning', 'Conclusions'])
#--------------------Introducción----------------------------#

if selection == 'Introduction':
    
    st.markdown("<h1 style='text-align: center; font-size: 45px; font-family: Beaufort;'>INTRODUCTION</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([.2,2,.2])
    with col2:
        st.write('<b>League of legends</b>(LoL) is a MOBA (Multiplayer Online Battle Arena) video game developed and distributed by Riot games and released in 2009.', unsafe_allow_html=True)
        st.write('''This type of game is based on competition between two teams to achieve a goal. As for LoL, the games are made up of 2 teams of 5 players,
                Each player has a different role and chooses a character (champion) to play the game with.
                The ultimate goal of the game is to destroy the enemy base while defending your own.''', unsafe_allow_html=True)
        st.write('''In terms of its impact, LoL has been and still is one of the most important and most played games in the world,
                 with leagues in many countries and its international competitions are among the most watched.''', unsafe_allow_html=True)
    
        st.write('')
        st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
        st.write('')

        montly_players = pd.DataFrame()
        montly_players['year'] = ['2011', '2012', '2014', '2017', '2018',  '2019', '2020', '2021', '2022']
        montly_players['players'] = [11.5, 32, 65, 100, 75, 117, 137, 149, 152]

        fig = px.histogram(montly_players, x= 'year', y = 'players',
                    template='plotly_dark',
        
        )
        fig.update_layout(title='<b>Monthly average players by year(millions)<b>', 
                    font_family="Spiegel",
                    titlefont={'size': 30},
                    showlegend=False,
                    )

        colors= ['#0AC8B9']*10
        fig.update_traces(marker_color=colors,
                    marker_line_width=2.5)
        fig.update_layout(width=800, height=500)

        st.plotly_chart(fig)

        st.write('')
        st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
        st.write('')
        st.subheader('Region - Europe West')

        st.image(image_region, caption='EUW region, Source: https://www.lolfinity.com/eune-players-who-are-pros-in-other-regions/', width=900)

        st.write('')
        st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
        st.write('')

        st.subheader('Map - Summoners Rift')

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_map, width=460, caption='Summoners Rift minimap')
        with col2:
            st.image(image_map2, width= 550, caption='Summoners Rift map')
    
        st.write('')
        st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
        st.write('')

        st.subheader('Roles')

        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(image_top, width=100, caption='Top')
        with col2:
            st.image(image_jungle, width= 94, caption='Jungle')
        with col3:
            st.image(image_mid, width= 100, caption='Mid')
        with col4:
            st.image(image_adc, width= 100, caption='Adc')
        with col5:
            st.image(image_support, width= 100, caption='Support')

        st.write('')
        st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
        st.write('')

        st.subheader('Leagues')

        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
        with col1:
            st.image(image_iron, width=100, caption='IRON')
        with col2:
            st.image(image_bronze, width= 100, caption='BRONZE')
        with col3:
            st.image(image_silver, width= 100, caption='SILVER')
        with col4:
            st.image(image_gold, width= 100, caption='GOLD')
        with col5:
            st.image(image_plat, width= 100, caption='PLATINUM')
        with col6:
            st.image(image_dia, width= 100, caption='DIAMOND')
        with col7:
            st.image(image_master, width= 100, caption='MASTER')
        with col8:
            st.image(image_grandmaster, width= 100, caption='GRANDMASTER')
        with col9:
            st.image(image_challenger, width= 100, caption='CHALLENGER')
    

    
#--------------------Data Collection----------------------------#

if selection == 'Data collection':
    st.header('Data collection')
    st.subheader('Players data')

    st.write('', unsafe_allow_html=True)

    st.markdown(f'[Link API League of Legends](https://developer.riotgames.com/)')

    st.code("""# Variables

server = 'euw1'
game_type = 'RANKED_SOLO_5x5'
league = ['DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
division = ['I', 'II', 'III', 'IV']
page = 1
api_key = 'X' """)
    
    st.code("""# Función para obtener la url, dependiendo de varios parametros.
            
def url_players(league, division, page):
url = f"https://{server}.api.riotgames.com/lol/league-exp/v4/entries/{game_type}/{league}/{division}?page={page}&api_key={api_key}"
return url""")


    st.code('''# Función para obtener los datos de todas las páginas de jugadores, decidiendo liga y división y teniendo en cuenta el tiempo de espera
    
    def get_players_data_3(league, division, start_page):
    all_data = [] # Creamos esta variable para que agrupe los datos de cada página
    counter = 0  # Contador para llevar el seguimiento de las solicitudes realizadas a la API
                 # con el objetivo de que no se pare la ejecución por llegar al límite
    # Comenzamos el bucle para que obtenga la información de todas las páginas por liga y división
    while True:
        url = url_players(league, division, start_page) # traemos el link de la función anterior
        data = requests.get(url).json() # obtenemos la información en formato json
        
        if not data:  # este condicional le dice al while cuando no hay datos en la página por lo tanto acaba de buscar
            break
        
        all_data.extend(data) # Mientras el bucle se ejecuta va añadiendo toda la información que se obtiene por página en la variable all_data
        counter += 1 # Aumenta el contador en 1 
        
        # Agregamos el retraso de 1 minuto cada 50 páginas (límite 100 cada 2 minutos)
        if counter % 50 == 0:
            time.sleep(60)  # Retraso de 1 minuto
        
        start_page += 1
    
    return all_data''')

    st.write('<b>Los datos obtenidos se agruparon por divisiones, ligas y el conjunto entero  </b>', unsafe_allow_html=True)
    st.table(players.head())

    st.write('')
    st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
    st.write('')

    st.header('Match data')
    st.subheader('First part')
    st.write('Raw data')
    st.code(r'''# 1 - Todo junto

league = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\players\gold\gold_2')
league_names = league['summonerName']
league_names_list = list(league_names) # creamos la lista con todos los players 

random_league_names = random.sample(league_names_list, 200) # Establecemos el número de players

# 2 - Puuid

from urllib.parse import quote

def url_puuid(name2):
    encoded_name = quote(name2)
    url = f"https://{server}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{encoded_name}?api_key={api_key}"
    return url

def get_players_puuid4(lista):

    all_data = []

    for name in lista:
        url = url_puuid(name)  # Obtener la URL para el nombre actual
        response = requests.get(url)

        if response.status_code == 429:  # Se ha excedido el límite de solicitudes
            print("1Esperando 1 minutos...")
            time.sleep(60)  # Esperar 2 minutos antes de realizar la siguiente solicitud
            response = requests.get(url)  # Realizar la solicitud nuevamente

        data = response.json()
        all_data.append(data)

    return all_data

puuid_league_names = get_players_puuid4(random_league_names) # usamos la función

puuid_league_names_df = pd.DataFrame(puuid_league_names) # lo convertimos a df

puuid_df = puuid_league_names_df['puuid'] # nos quedamos con la columna puuid para usarla más adelante


# 3 - Match

def url_puuid(puuid):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count=6&api_key={api_key}"
    return url

def get_match_id(lista):

    all_data = []

    for puuid in lista:
        url = url_puuid(puuid)  # Obtener la URL para el nombre actual
        response = requests.get(url)

        if response.status_code == 429:  # Se ha excedido el límite de solicitudes
            print("2Esperando 1 minutos...")
            time.sleep(60)  # Esperar 2 minutos antes de realizar la siguiente solicitud
            response = requests.get(url)  # Realizar la solicitud nuevamente

        data = response.json()
        all_data.append(data)

    return all_data

matches_id = get_match_id(puuid_df)

matches_id = list(itertools.chain.from_iterable(matches_id)) # convertimos la lista de listas, en una lista única
'''
)
    st.code('''# 4 - Match info

def url_matches(match_id):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}"
    return url

def get_match_data(lista):

    all_data = []
   
    for match_id in lista:
        url = url_matches(match_id)  # Obtener la URL para el nombre actual
        response = requests.get(url)

        if response.status_code == 429:  # Se ha excedido el límite de solicitudes
            print("3Esperando 2 minutos...")
            time.sleep(120)  # Esperar 2 minutos antes de realizar la siguiente solicitud
            response = requests.get(url)

        data = response.json()
        all_data.append(data)

    return all_data
            
data_match_challenger = get_match_data(matches_id)
'''
)

    st.write('')
    st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
    st.write('')

    st.subheader('Second part')
    st.write('Column choice and conversion to dataframe')
    st.code('''def clean_matches(lista):
    prueba = pd.DataFrame()
    columnas = ['gameId','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald',
                
        't1_champ1id','t1_champ1_sum1','t1_champ1_sum2','t1_champ2id','t1_champ2_sum1','t1_champ2_sum2','t1_champ3id','t1_champ3_sum1','t1_champ3_sum2',
        't1_champ4id','t1_champ4_sum1','t1_champ4_sum2','t1_champ5id','t1_champ5_sum1','t1_champ5_sum2',
        
        't1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills', 't1_riftHeraldKills',
        't1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5',
        
        't2_champ1id','t2_champ1_sum1','t2_champ1_sum2','t2_champ2id','t2_champ2_sum1','t2_champ2_sum2','t2_champ3id','t2_champ3_sum1','t2_champ3_sum2',
        't2_champ4id','t2_champ4_sum1','t2_champ4_sum2','t2_champ5id','t2_champ5_sum1','t2_champ5_sum2',
        
        't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills',
        't2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5', '','']
    loc = 0
    col = 0

#--------------------------------------------------------------#

    # MATCH DATA

    # MATCHID
    all_data = []
    for match in lista:
        datos = match['metadata']['matchId']
        all_data.append(datos)
    all_data = pd.Series(all_data)
    prueba.insert(loc=loc, column=columnas[col], value=all_data)

    loc+=1
    col+=1
    # GAMEDURATION
    all_data = []
    for match in lista:
        datos = match['info']['gameDuration']
        all_data.append(datos)
    all_data = pd.Series(all_data)
    prueba.insert(loc=loc, column=columnas[col], value=all_data)) '''
)
    st.write('')

    st.table(matches.head())

    st.subheader('Champions data')
    st.write('Obtención de los datos de los campeones')
    st.code('''champions_data = pd.read_json('http://ddragon.leagueoflegends.com/cdn/13.13.1/data/en_US/champion.json')''')


#--------------------Preprocesamiento----------------------------#
if selection == 'Preprocessing':

    st.header('Preprocessing')
    st.subheader('Players')
    st.write('', unsafe_allow_html=True)
    st.code(r'''players_df = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\players\players.csv')''')
    st.code('''df = players_df.drop(['miniSeries', 'queueType'], axis=1)''')
    st.code('''# Diccionario para mapear la columna ligas
    mapping_tier = {'IRON': '0', 'BRONZE': '1', 'SILVER': '2', 'GOLD': '3', 'PLATINUM': '4', 'DIAMOND': '5', 'MASTER': '6', 'GRANDMASTER': '7', 'CHALLENGER': '8'}

    # Aplicamos el mapeo a la columna de ligas
    df['tier'] = df['tier'].map(mapping_tier)''')

    st.code('''# Diccionario para mapear la columna divisiones
    mapping_rank = {'I': '1', 'II': '2', 'III': '3', 'IV': '4'}

   # Aplicamos el mapeo a la columna de divisiones    
    df['rank'] = df['rank'].map(mapping_rank)  ''')

    st.code('''df[df['summonerName'] == 'Aizzo'] .sort_values(by=['tier', 'rank', 'leaguePoints'], ascending=[False, True, False])''')
    st.code('''df = df.sort_values(by=['tier', 'rank', 'leaguePoints'], ascending=[False, True, False])''')
    st.code('''df = df.drop_duplicates('summonerName')''')
    
    st.code('''df = df.drop(['leagueId', 'summonerId', 'summonerName'], axis=1)''')
    st.code('''# True = Yes

    mapping_firsthBlood = {True: 1, False: 0}
    mapping_hotStreak = {True: 1, False: 0}
    mapping_veteran = {True: 1, False: 0}
    mapping_inactive = {True: 1, False: 0}

    df['firstBlood'] = df['firstBlood'].map(mapping_firsthBlood)
    df['hotStreak'] = df['hotStreak'].map(mapping_hotStreak)
    df['veteran'] = df['veteran'].map(mapping_veteran)
    df['inactive'] = df['inactive'].map(mapping_inactive)''')

    st.code('''df['division'] = df['rank']
    df['league'] = df['tier']''')
    st.code('''df = df[['league', 'division', 'wins', 'losses', 'leaguePoints', 'hotStreak', 'firstBlood', 'veteran', 'inactive']]''')
    st.code('''df = df.reset_index()''')
    st.table(players_clean.head())


    st.subheader('Matches')
    st.write('', unsafe_allow_html=True)

    st.write('Drop matches that lasted less than 15 minuts', unsafe_allow_html=True)
    st.code('''matches_before15 = matches.loc[matches['gameDuration'] <= 900]''')
    
    st.write('Change seasonId to make it more readable', unsafe_allow_html=True)
    st.code('''
    # split the seasonId by '.', keep the first 2 numbers and then join them by a '.'
    matches['seasonId'] = matches['seasonId'].str.split('.').str[:2].str.join('.')''')

    st.write('Codificamos las columnas principales', unsafe_allow_html=True)
    st.code('''
    mapping_ft = {True: 1, False: 2}
    matches['firstTower'] = matches['firstTower'].map(mapping_ft)
    
    mapping_ft = {True: 1, False: 2, 0: 0}
    matches['firstInhibitor'] = matches['firstInhibitor'].map(mapping_ft)
   
    mapping_fba = {True: 1, False: 2, 0: 0}
    matches['firstBaron'] = matches['firstBaron'].map(mapping_fba)
    
    mapping_fd = {True: 1, False: 2, 0: 0}
    matches['firstDragon'] = matches['firstDragon'].map(mapping_fd)
    
    mapping_fh = {True: 1, False: 2, 0: 0}
    matches['firstRiftHerald'] = matches['firstRiftHerald'].map(mapping_fh)
    
    mapping_w = {True: 1, False: 2}
    matches['winner'] = matches['winner'].map(mapping_w)''')

    st.code('''matches.drop_duplicates(keep='first', inplace=True)''')
    
    st.write('Comprobamos duplicados y eliminamos', unsafe_allow_html=True)
    st.code('''matches_dupli = matches['gameId'].value_counts()''')
    st.code('''matches.drop_duplicates(keep='first', inplace=True)''')

    st.write('Cambiamos a minutos la duración', unsafe_allow_html=True)
    st.code('''matches['gameDuration'] = (matches['gameDuration']/60)''')
    st.code('''matches['gameDuration'] = matches['gameDuration'].round()''')
    st.code('''matches['gameDuration'] = matches['gameDuration'].astype(int)''')
    matches_clean = matches_clean.drop('Unnamed: 0', axis=1)
    st.table(matches_clean.head())

#--------------------EDA----------------------------#
if selection == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')    

#------------------------------------------------#

    st.image(players_league)

#------------------------------------------------#
    col1,col2 = st.columns(2)
    with col1:
        width = 110
        side = 40
        _, container, _ = st.columns([side, width, side]) # chatgpt
    container.video(data=VIDEO_DATA2)
    with col2:
        custom_palette = ['#262626',  '#CD7F32','#B5B7BB'
                    ,  '#FFD700',  '#E5E4E2', '#6DCEFC',
                    '#90269B', '#C10A0A', '#33E0FB']

        fig = px.histogram(winrate_league, x= 'league', y = 'winrate',
                        color='league',
                        color_discrete_sequence=custom_palette,
                        template='plotly_dark',
            
        )
        fig.update_layout(title='<b>Mean winrate by league<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        showlegend=False,
                        )

        fig.update_traces(marker_line_width=2.5)

        fig.update_yaxes(range=[25,61], dtick=10)
        fig.update_layout(width=800, height=400)
        for i, winrate in enumerate(winrate_league['winrate']):
            fig.data[i].text = f'{winrate:.2f}%'  
            fig.data[i].textposition = 'outside'

        st.plotly_chart(fig)

#------------------------------------------------#


#------------------------------------------------#

    df_matches = matches.drop(['gameId', 'seasonId'], axis=1)

#------------------------------------------------#
    df_matches = matches_clean.drop(matches_clean[matches_clean['gameDuration'] >= 60].index)

# Game Duration
#------------------------------------------------#
    col1,col2 = st.columns(2)
    with col2:
        st.image(champ_ryze, width=715)
    with col1:
        df_matches = df_matches.drop(df_matches[df_matches['gameDuration'] >= 60].index)
        df_matches = df_matches.drop(df_matches[df_matches['gameDuration'] <= 16].index)
        fig = px.histogram(df_matches, x = 'gameDuration',
                        histnorm='percent',
                        template = 'plotly_dark',
                    )

        fig.update_layout(title='<b>Game duration<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#0AC8B9']*45


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=800, height=500)

        st.plotly_chart(fig)

        st.write('The average duration is:', (df_matches['gameDuration'].sum()/len(df_matches)).round(2))
#------------------------------------------------#
    col1, col2, col3 = st.columns(3)
    with col1:

        iron_matches_duration = iron_matches_duration.drop(iron_matches_duration[iron_matches_duration['gameDuration'] >= 60].index)
        iron_matches_duration = iron_matches_duration.drop(iron_matches_duration[iron_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(iron_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Iron<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#8A8A8A']*60


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (iron_matches_duration['gameDuration'].sum()/len(iron_matches_duration)).round(2))
       
    #------------------------------------------------#

    with col2:
        bronze_matches_duration = bronze_matches_duration.drop(bronze_matches_duration[bronze_matches_duration['gameDuration'] >= 60].index)
        bronze_matches_duration = bronze_matches_duration.drop(bronze_matches_duration[bronze_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(bronze_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Bronze<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#CD7F32']*60
        fig.update_layout(width=550, height=400)

        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)

        st.plotly_chart(fig)

        st.write('The average duration is:', (bronze_matches_duration['gameDuration'].sum()/len(bronze_matches_duration)).round(2))

    with col3:
        silver_matches_duration = silver_matches_duration.drop(silver_matches_duration[silver_matches_duration['gameDuration'] >= 60].index)
        silver_matches_duration = silver_matches_duration.drop(silver_matches_duration[silver_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(silver_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Silver<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#B5B7BB']*60


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (silver_matches_duration['gameDuration'].sum()/len(silver_matches_duration)).round(2))
#------------------------------------------------#
    col1, col2, col3 = st.columns(3)
    with col1:

        gold_matches_duration = gold_matches_duration.drop(gold_matches_duration[gold_matches_duration['gameDuration'] >= 60].index)
        gold_matches_duration = gold_matches_duration.drop(gold_matches_duration[gold_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(gold_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Gold<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#FFD700']*60


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (gold_matches_duration['gameDuration'].sum()/len(gold_matches_duration)).round(2))
    
    #------------------------------------------------#

    with col2:
        plat_matches_duration = plat_matches_duration.drop(plat_matches_duration[plat_matches_duration['gameDuration'] >= 60].index)
        plat_matches_duration = plat_matches_duration.drop(plat_matches_duration[plat_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(plat_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Platinum<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#E5E4E2']*60


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (plat_matches_duration['gameDuration'].sum()/len(plat_matches_duration)).round(2))

    with col3:
        dia_matches_duration = dia_matches_duration.drop(dia_matches_duration[dia_matches_duration['gameDuration'] >= 60].index)
        dia_matches_duration = dia_matches_duration.drop(dia_matches_duration[dia_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(dia_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Diamond<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#6DCEFC']*60


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (dia_matches_duration['gameDuration'].sum()/len(dia_matches_duration)).round(2))
#------------------------------------------------#

    col1, col2, col3 = st.columns(3)
    with col1:

        master_matches_duration = master_matches_duration.drop(master_matches_duration[master_matches_duration['gameDuration'] >= 60].index)
        master_matches_duration = master_matches_duration.drop(master_matches_duration[master_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(master_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Master<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#90269B']*70


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (master_matches_duration['gameDuration'].sum()/len(master_matches_duration)).round(2))
    
    #------------------------------------------------#

    with col2:
        grandmaster_matches_duration = grandmaster_matches_duration.drop(grandmaster_matches_duration[grandmaster_matches_duration['gameDuration'] >= 60].index)
        grandmaster_matches_duration = grandmaster_matches_duration.drop(grandmaster_matches_duration[grandmaster_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(grandmaster_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Grandmaster<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#C10A0A']*70


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (grandmaster_matches_duration['gameDuration'].sum()/len(grandmaster_matches_duration)).round(2))
    
    with col3:
        challenger_matches_duration = challenger_matches_duration.drop(challenger_matches_duration[challenger_matches_duration['gameDuration'] >= 60].index)
        challenger_matches_duration = challenger_matches_duration.drop(challenger_matches_duration[challenger_matches_duration['gameDuration'] <= 16].index)

        fig = px.histogram(challenger_matches_duration, x = 'gameDuration',
                            histnorm='percent',
                            template = 'plotly_dark',
                        )

        fig.update_layout(title='<b>Game duration - Challenger<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )

        colors = ['#33E0FB']*70


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)
        fig.update_layout(width=550, height=400)

        st.plotly_chart(fig)

        st.write('The average duration is:', (challenger_matches_duration['gameDuration'].sum()/len(challenger_matches_duration)).round(2))
#------------------------------------------------#
    iron_matches_duration_mean = (iron_matches_duration['gameDuration'].sum()/len(iron_matches_duration)).round(2)
    bronze_matches_duration_mean = (bronze_matches_duration['gameDuration'].sum()/len(bronze_matches_duration)).round(2)
    silver_matches_duration_mean = (silver_matches_duration['gameDuration'].sum()/len(silver_matches_duration)).round(2)
    gold_matches_duration_mean = (gold_matches_duration['gameDuration'].sum()/len(gold_matches_duration)).round(2)
    plat_matches_duration_mean = (plat_matches_duration['gameDuration'].sum()/len(plat_matches_duration)).round(2)
    dia_matches_duration_mean = (dia_matches_duration['gameDuration'].sum()/len(dia_matches_duration)).round(2)
    master_matches_duration_mean = (master_matches_duration['gameDuration'].sum()/len(master_matches_duration)).round(2)
    grandmaster_matches_duration_mean = (grandmaster_matches_duration['gameDuration'].sum()/len(grandmaster_matches_duration)).round(2)
    challenger_matches_duration_mean = (challenger_matches_duration['gameDuration'].sum()/len(challenger_matches_duration)).round(2)

    gameduration_means = pd.DataFrame()

    gameduration_means['league'] = ['iron', 'bronze', 'silver', 'gold', 'plat', 'dia', 'master', 'grandmaster', 'challenger']

    gameduration_means['mean_duration'] = [iron_matches_duration_mean, bronze_matches_duration_mean, silver_matches_duration_mean,
                                        gold_matches_duration_mean, plat_matches_duration_mean, dia_matches_duration_mean,
                                        master_matches_duration_mean, grandmaster_matches_duration_mean, challenger_matches_duration_mean]

    gameduration_means = gameduration_means[['league', 'mean_duration']]

    col1,col2 = st.columns(2)
    with col1:
        width = 100
        side = 50
        _, container, _ = st.columns([side, width, side]) # chatgpt
    container.video(data=VIDEO_DATA1)
    with col2:
        fig = px.histogram(gameduration_means, x = 'league', y = 'mean_duration',
                        template = 'plotly_dark',
                    )

        fig.update_layout(title='<b>Game duration by league<b>', 
                        font_family="Spiegel",
                        titlefont={'size': 30},
                        
                        )
        fig.update_yaxes(range=[25, 31.5])
        fig.update_layout(yaxis_title="Mean Game duration")

        colors = ['#8A8A8A', '#CD7F32', '#B5B7BB', '#FFD700', '#E5E4E2', '#6DCEFC', '#90269B', '#C10A0A', '#33E0FB']


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)

        st.plotly_chart(fig)

#------------------------------------------------#
    col1,col2 = st.columns(2)
    with col2:
        st.image(champ_riven, width=715)

    with col1:
        win_by_team = df_matches['winner'].value_counts()

        fig = px.bar(win_by_team,
                template = 'plotly_dark',
                )

        fig.update_layout(title='<b>WINS BY TEAM<b>',
                        font_family="Spiegel",
                        titlefont={'size': 30, 'color':'#F0E6D2'},
                        showlegend=False
                        )

        colors = ['#1F8CFF', '#FF4040']


        fig.update_traces(marker_color=colors, marker_line_color=None,
                        marker_line_width=2.5, opacity=None)

        fig.update_xaxes(
            ticktext=['Team 1', 'Team 2'],
            tickvals=[1, 2],

        )

        st.plotly_chart(fig)
#------------------------------------------------#
    cond1 = df_matches['firstBlood'] == 1
    cond2 = df_matches['firstBlood'] == 2
    firstblood1 = df_matches.loc[cond1, 'winner'].value_counts()
    firstblood2 = df_matches.loc[cond2, 'winner'].value_counts()

    firstblood1 = pd.DataFrame(firstblood1)
    firstblood2 = pd.DataFrame(firstblood2)

    total_firstblood1 = (firstblood1['winner'][1])+(firstblood1['winner'][2])
    total_firstblood2 = (firstblood2['winner'][1])+(firstblood2['winner'][2])

    firstblood1['percentage'] = (firstblood1['winner'][1])/total_firstblood1
    firstblood2['percentage'] = (firstblood2['winner'][1])/total_firstblood2

    firstblood1['percentage'][1] = (firstblood1['winner'][1])/total_firstblood1
    firstblood2['percentage'][1] = (firstblood2['winner'][1])/total_firstblood2

    firstblood1['percentage'][2] = (firstblood1['winner'][2])/total_firstblood1
    firstblood2['percentage'][2] = (firstblood2['winner'][2])/total_firstblood2

    col1,col2 = st.columns(2)
    with col1:
        width = 100
        side = 50
        _, container, _ = st.columns([side, width, side]) # chatgpt
    container.video(data=VIDEO_DATA3)
    with col2:
        fig1 = px.bar(firstblood1,
                    template = 'plotly_dark',
                    labels=firstblood1['percentage']
                    )


        colors = ['#1F8CFF', '#FF4040']


        fig1.update_traces(marker_color=colors)

        percentages1 = (firstblood1['percentage'] * 100).round(4).astype(str) + '%'
        fig1.update_traces(text=percentages1, textposition='outside')

        fig2 = px.bar(firstblood2,
                    template = 'plotly_dark',
                    )



        colors = ['#FF4040', '#1F8CFF' ]


        fig2.update_traces(marker_color=colors)

        percentages2 = (firstblood2['percentage'] * 100).round(4).astype(str) + '%'
        fig2.update_traces(text=percentages2, textposition='outside')



        # Make subplots
        fig = make_subplots(rows=1, cols=2)

        # Graphics on the subplots
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)

        # Update design
        fig.update_layout(showlegend=False, title=f'<b>WIN % BY FIRSTBLOOD TEAM<b>', 
                        font_family="beaufort",
                        titlefont={'size': 30, 'color':'#F0E6D2'},
                        template='plotly_dark')

        fig.update_xaxes(
            ticktext=['Team 1', 'Team 2'],
            tickvals=[1, 2],
            row=1, col=1
        )
        fig.update_xaxes(
            ticktext=['Team 1', 'Team 2'],
            tickvals=[1, 2],
            row=1, col=2
        )

        # Subtitles

        fig.update_layout(  
            annotations=[
                dict(
                    text="<b>TEAM 1 FIRSTBLOOD<b>",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.1,
                    showarrow=False,
                    font=dict(size=20, color="#C89B3C")
                ),
                dict(
                    text="<b>TEAM 2 FIRSTBLOOD<b>",
                    xref="paper",
                    yref="paper",
                    x=0.65,
                    y=1.1,
                    showarrow=False,
                    font=dict(size=20, color="#C89B3C")
                )
            ]
        )
        fig.update_layout(width=800, height=500)

        st.plotly_chart(fig)

#------------------------------------------------#
    st.markdown('-------------------------------------------------------------------------------------', unsafe_allow_html=True)

    champion_name = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\champions\champions')
    champion_name = champion_name[['Name', 'Id']]
#------------------------------------------------#
    st.header('Champions')

    col1,col2,col3 = st.columns(3)
    with col1:

        st.image(image_iron, caption='IRON')   
        st.subheader('TOP')
        st.table(iron_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(iron_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(iron_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(iron_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(iron_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col2:
        st.image(image_bronze, caption='BRONZE')
        st.subheader('TOP')
        st.table(bronze_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(bronze_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(bronze_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(bronze_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(bronze_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col3:
        st.image(image_silver, caption='SILVER')
        st.subheader('TOP')
        st.table(silver_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(silver_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(silver_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(silver_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')

        st.table(silver_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))

#------------------------------------------------#

    col1,col2,col3 = st.columns(3)
    with col1:

        st.image(image_gold, caption='GOLD')   
        st.subheader('TOP')
        st.table(gold_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(gold_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(gold_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(gold_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(gold_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col2:
        st.image(image_plat, caption='PLATINUM')   
        st.subheader('TOP')
        st.table(plat_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(plat_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(plat_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(plat_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(plat_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col3:
        st.image(image_dia, caption='DIAMOND')   
        st.subheader('TOP')
        st.table(dia_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(dia_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(dia_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(dia_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(dia_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
#------------------------------------------------#
    col1,col2,col3 = st.columns(3)
    with col1:

        st.image(image_master, caption='MASTER')   
        st.subheader('TOP')
        st.table(master_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(master_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(master_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(master_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(master_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col2:
        st.image(image_grandmaster, caption='GRANDMASTER')   
        st.subheader('TOP')
        st.table(grandmaster_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(grandmaster_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(grandmaster_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(grandmaster_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(grandmaster_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
    with col3:
        st.image(image_challenger, caption='CHALLENGER')   
        st.subheader('TOP')
        st.table(challenger_matches_top_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('JUNGLE')
        st.table(challenger_matches_jungle_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('MID')
        st.table(challenger_matches_mid_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('ADC')
        st.table(challenger_matches_adc_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
        st.subheader('SUPPORT')
        st.table(challenger_matches_support_champions_winratio.drop(['Unnamed: 0', 'Id'],axis=1))
#------------------------------------------------#
    st.markdown('-------------------------------------------------------------------------------------', unsafe_allow_html=True)


    st.header('Machine Learning')
    st.code('''from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separar las características (X) y el objetivo (y)
X = df_matches_ml[['gameDuration', 't1_towerKills', 't2_towerKills', 't1_inhibitorKills', 't2_inhibitorKills', 't1_dragonKills', 't2_dragonKills', 't1_baronKills', 't2_baronKills']]
y = df_matches_ml['winner']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo de clasificación
model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)

# Calcular las probabilidades de pertenencia a cada clase
probabilities = model.predict_proba(X_test_scaled)

# Obtener las probabilidades para el equipo 1 y el equipo 2
prob_team1 = probabilities[:, 0]
prob_team2 = probabilities[:, 1]
''')
#------------------------------------------------#


#--------------------Machine Learning----------------------------#

if selection == 'Machine Learning':


    st.header('Machine Learning')
    model = load_model(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\lol_ml_gbc')
    df_matches_ml = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_Final\data\df_matches_ml')
    df_matches_ml = df_matches_ml.drop('Unnamed: 0', axis=1)

    st.title('')
    st.write('Select the variables to obtain the prediction.')


    # Separar las características (X) y el objetivo (y)
    X = df_matches_ml[['gameDuration', 't1_towerKills', 't2_towerKills', 't1_inhibitorKills', 't2_inhibitorKills', 't1_dragonKills', 't2_dragonKills', 't1_baronKills', 't2_baronKills']]
    y = df_matches_ml['winner']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar el modelo de clasificación
    model = GradientBoostingClassifier()
    model.fit(X_train_scaled, y_train)

    # Calcular las probabilidades de pertenencia a cada clase
    probabilities = model.predict_proba(X_test_scaled)

    # Obtener las probabilidades para el equipo 1 y el equipo 2
    prob_team1 = probabilities[:, 0]
    prob_team2 = probabilities[:, 1]

# Streamlit ML

    # Introducimos los controles en en streamlit para darle diferentes opciones
    game_duration = st.slider('gameDuration', min_value=0, max_value=60, value=0)
    t1_tower_kills = st.slider('t1_towerKills', min_value=0, max_value=11, value=0)
    t2_tower_kills = st.slider('t2_towerKills', min_value=0, max_value=11, value=0)
    t1_inhibitor_kills = st.slider('t1_inhibitorKills', min_value=0, max_value=11, value=0)
    t2_inhibitor_kills = st.slider('t2_inhibitorKills', min_value=0, max_value=11, value=0)
    t1_dragon_kills = st.slider('t1_dragonKills', min_value=0, max_value=7, value=0)
    t2_dragon_kills = st.slider('t2_dragonKills', min_value=0, max_value=7, value=0)
    t1_baron_kills = st.slider('t1_baronKills', min_value=0, max_value=5, value=0)
    t2_baron_kills = st.slider('t2_baronKills', min_value=0, max_value=5, value=0)


    # Mostrar el botón "Calcular Probabilidad"
    if st.button('Calculate results'):
            # Creamos un df con las variables que vamos a introducir
        data = pd.DataFrame({
            'gameDuration': [game_duration],
            't1_towerKills': [t1_tower_kills],
            't2_towerKills': [t2_tower_kills],
            't1_inhibitorKills': [t1_inhibitor_kills],
            't2_inhibitorKills': [t2_inhibitor_kills],
            't1_dragonKills': [t1_dragon_kills],
            't2_dragonKills': [t2_dragon_kills],
            't1_baronKills': [t1_baron_kills],
            't2_baronKills': [t2_baron_kills]
        })

            # Escalamos los datos
        scaler = StandardScaler()
        scaler.fit(X)

            # Escalamos los datos nuevos
        data_scaled = scaler.transform(data)

            # Calculamos las probabilidades
        probabilities = model.predict_proba(data_scaled)
        prob_team1 = probabilities[0, 0]
        prob_team2 = probabilities[0, 1]

            # Mostramos la probabilidad
        st.write('Probabilidad de que el equipo 1 gane:', (prob_team1*100).round(2),'%')
        st.write('Probabilidad de que el equipo 2 gane:', (prob_team2*100).round(2),'%')





#--------------------Conclusiones----------------------------#
if selection == 'Conclusions':
    
    st.header('Conclusion')

    st.write('''This analysis shows useful data for players who want to keep improving, so they can make better decisions,
              both to know which champions are performing better, and to have a guide within the games, about which objectives are more important.''')
    st.write('About 80% of players are grouped in bronze, silver and gold.')
    st.write('As the league increases, the winrate increases and the duration of the games decreases. ')
    st.write('There are no differences in terms of team.')

    st.subheader('Limitations and Future Analysis')
    st.write('<b>API Restrictions</b>: Only make 100 request every 2 minuts.', unsafe_allow_html=True)
    st.write('<b>Machine learning limitations</b>: A more in-depth analysis is require in order to make the ML more precise', unsafe_allow_html=True)


    st.write('')
    st.write('-------------------------------------------------------------------------------------', unsafe_allow_html=True)
    st.write('')

    st.header('References')
    st.write('Wiki: https://leagueoflegends.fandom.com/')
    st.write('Info https://activeplayer.io/league-of-legends/#:~:text=League%20of%20Legends%20Player%20Count%20(Monthly%20Active%20Players)&text=League%20of%20Legends%20has%20a,10%20%E2%80%93%2011%20active%20players%20daily.')
    st.write('Info https://www.leagueoflegends.com/es-es/how-to-play/?_gl=1*png51g*_ga*MTU2NzUyNDU1NS4xNjg5NTkzMDY4*_ga_FXBJE5DEDD*MTY4OTU5MzA2OC4xLjEuMTY4OTU5MzA5NC4zNC4wLjA.')
    st.write('Graph https://headphonesaddict.com/wp-content/uploads/2022/11/Number-of-monthly-LoL-players-per-years.png.webp')
    st.write('Info https://activeplayer.io/league-of-legends/')
    st.write('Wiki: https://leagueoflegends.fandom.com/')
    st.write('Info https://escharts.com/')
    st.write('https://www.ggrecon.com/guides/league-of-legends-rank-distribution/')
    st.write('https://www.leagueofgraphs.com/rankings/rank-distribution/euw')



st.sidebar.markdown('-------------------------------------------------------------------------------------', unsafe_allow_html=True)

st.sidebar.image(legal, width=50)



# theme = primaryColor="#c89b3c"backgroundColor="#010a13" secondaryBackgroundColor="#010a13" textColor="#f0e6d2"


