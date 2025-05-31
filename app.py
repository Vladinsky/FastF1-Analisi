from flask import Flask, render_template, request, jsonify
import fastf1
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Abilita cache FastF1
fastf1.Cache.enable_cache('cache')

# Dizionari dei dati
piloti_per_anno = {
    2015: ["HAM", "ROS", "VET", "RAI", "BOT", "MAS", "ALO", "BUT", "RIC", "KVY", "PER", "HUL", "GRO", "MAL", "SAI", "VER", "ERI", "NAS"],
    2016: ["HAM", "ROS", "VET", "RAI", "BOT", "MAS", "ALO", "BUT", "RIC", "VER", "PER", "HUL", "GRO", "GUT", "SAI", "KVY", "ERI", "NAS", "WEH", "OCO"],
    2017: ["VET", "HAM", "BOT", "RAI", "VER", "RIC", "PER", "OCO", "HUL", "SAI", "STR", "MAS", "ALO", "VAN", "GRO", "MAG", "ERI", "WEH"],
    2018: ["HAM", "VET", "RAI", "BOT", "VER", "RIC", "PER", "OCO", "HUL", "SAI", "LEC", "ERI", "GRO", "MAG", "ALO", "VAN", "STR", "SIR"],
    2019: ["HAM", "BOT", "VET", "LEC", "VER", "GAS", "ALB", "RIC", "HUL", "SAI", "PER", "STR", "RAI", "GIO", "GRO", "MAG", "KUB", "RUS"],
    2020: ["HAM", "BOT", "VER", "ALB", "LEC", "VET", "PER", "STR", "RIC", "OCO", "SAI", "NOR", "RAI", "GIO", "GRO", "MAG", "RUS", "LAT"],
    2021: ["VER", "HAM", "BOT", "PER", "LEC", "SAI", "RIC", "NOR", "GAS", "TSU", "OCO", "ALO", "VET", "STR", "RAI", "GIO", "RUS", "LAT", "MSC", "MAZ"],
    2022: ["VER", "PER", "LEC", "SAI", "HAM", "RUS", "NOR", "RIC", "OCO", "ALO", "GAS", "TSU", "VET", "STR", "BOT", "ZHO", "MSC", "MAG", "ALB", "LAT"],
    2023: ["VER", "PER", "HAM", "RUS", "LEC", "SAI", "NOR", "PIA", "ALO", "STR", "GAS", "OCO", "ALB", "SAR", "BOT", "ZHO", "MAG", "HUL"],
    2024: ["VER", "PER", "HAM", "RUS", "LEC", "SAI", "NOR", "PIA", "ALO", "STR", "GAS", "OCO", "ALB", "SAR", "BOT", "ZHO", "MAG", "HUL", "RIC", "TSU"]
}

gp_per_anno = {
    2015: ["Australia", "Malaysia", "China", "Bahrain", "Spain", "Monaco", "Canada", "Austria", "Great Britain", "Hungary", "Belgium", "Monza", "Singapore", "Japan", "Russia", "Austin", "Mexico", "Brazil", "Abu Dhabi"],
    2016: ["Australia", "Bahrain", "China", "Russia", "Spain", "Monaco", "Canada", "Azerbaijan", "Austria", "Great Britain", "Hungary", "Germany", "Belgium", "Monza", "Singapore", "Malaysia", "Japan", "Austin", "Mexico", "Brazil", "Abu Dhabi"],
    2017: ["Australia", "China", "Bahrain", "Russia", "Spain", "Monaco", "Canada", "Azerbaijan", "Austria", "Great Britain", "Hungary", "Belgium", "Monza", "Singapore", "Malaysia", "Japan", "Austin", "Mexico", "Brazil", "Abu Dhabi"],
    2018: ["Australia", "Bahrain", "China", "Azerbaijan", "Spain", "Monaco", "Canada", "France", "Austria", "Great Britain", "Germany", "Hungary", "Belgium", "Monza", "Singapore", "Russia", "Japan", "Austin", "Mexico", "Brazil", "Abu Dhabi"],
    2019: ["Australia", "Bahrain", "China", "Azerbaijan", "Spain", "Monaco", "Canada", "France", "Austria", "Great Britain", "Germany", "Hungary", "Belgium", "Monza", "Singapore", "Russia", "Japan", "Mexico", "Austin", "Brazil", "Abu Dhabi"],
    2020: ["Austria", "Styria", "Hungary", "Great Britain", "70thAnniversary", "Spain", "Belgium", "Monza", "Tuscany", "Russia", "Eifel", "Portugal", "Imola", "Turkey", "Bahrain", "Sakhir", "Abu Dhabi"],
    2021: ["Bahrain", "Imola", "Portugal", "Spain", "Monaco", "Azerbaijan", "France", "Styria", "Austria", "Great Britain", "Hungary", "Belgium", "Zandvoort", "Monza", "Russia", "Turkey", "Austin", "Mexico", "Brazil", "Qatar", "Saudi Arabia", "Abu Dhabi"],
    2022: ["Bahrain", "Saudi Arabia", "Australia", "Imola", "Miami", "Spain", "Monaco", "Azerbaijan", "Canada", "Great Britain", "Austria", "France", "Hungary", "Belgium", "Zandvoort", "Monza", "Singapore", "Japan", "Austin", "Mexico", "Brazil", "Abu Dhabi"],
    2023: ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", "Monaco", "Spain", "Canada", "Austria", "Great Britain", "Hungary", "Belgium", "Zandvoort", "Monza", "Singapore", "Japan", "Qatar", "Austin", "Mexico", "Brazil", "LasVegas", "Abu Dhabi"],
    2024: ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Imola", "Monaco", "Canada", "Spain", "Austria", "Great Britain", "Hungary", "Belgium", "Zandvoort", "Monza", "Azerbaijan", "Singapore", "Austin", "Mexico", "Brazil", "LasVegas", "Qatar", "Abu Dhabi"]
}

def format_laptime(laptime):
    """Converte il tempo giro nel formato mm:ss:mmm"""
    if pd.isna(laptime) or laptime is None:
        return "N/A"
    
    try:
        # Se è già un timedelta
        if isinstance(laptime, pd.Timedelta):
            total_seconds = laptime.total_seconds()
        # Se è un datetime.timedelta
        elif isinstance(laptime, timedelta):
            total_seconds = laptime.total_seconds()
        # Se è un numero (secondi)
        elif isinstance(laptime, (int, float)):
            total_seconds = laptime
        else:
            # Prova a convertire
            total_seconds = float(laptime)
        
        # Calcola minuti, secondi e millisecondi
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        
        return f"{minutes}:{seconds:02d}:{milliseconds:03d}"
    
    except Exception as e:
        print(f"Errore nel formattare il tempo: {e}")
        return "N/A"

def rotate(xy, *, angle):
    """Ruota le coordinate di un angolo dato"""
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def get_circuit_plot(session):
    """Genera il grafico del circuito con numeri delle curve - VERSIONE CORRETTA"""
    try:
        # Ottieni i dati della posizione dal giro più veloce
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()
        circuit_info = session.get_circuit_info()
        
        # Ottieni le coordinate del tracciato
        track = pos.loc[:, ('X', 'Y')].to_numpy()
        
        # Converti l'angolo di rotazione da gradi a radianti
        track_angle = circuit_info.rotation / 180 * np.pi
        
        # Ruota il tracciato
        rotated_track = rotate(track, angle=track_angle)
        
        # Crea figura con dimensioni corrette
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
        
        # Traccia il circuito
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], 
                color='white', linewidth=4, solid_capstyle='round')
        
        # Vettore di offset per i numeri delle curve
        offset_vector = [500, 0]
        
        # Aggiungi i numeri delle curve
        for _, corner in circuit_info.corners.iterrows():
            # Crea il testo dal numero e lettera della curva
            txt = f"{corner['Number']}{corner['Letter']}"
            
            # Converti l'angolo da gradi a radianti
            offset_angle = corner['Angle'] / 180 * np.pi
            
            # Ruota il vettore di offset
            offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
            
            # Aggiungi l'offset alla posizione della curva
            text_x = corner['X'] + offset_x
            text_y = corner['Y'] + offset_y
            
            # Ruota la posizione del testo
            text_x, text_y = rotate([text_x, text_y], angle=track_angle)
            
            # Ruota il centro della curva
            track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)
            
            # Disegna un cerchio grigio
            ax.scatter(text_x, text_y, color='grey', s=140, zorder=10)
            
            # Disegna una linea dal tracciato al cerchio
            ax.plot([track_x, text_x], [track_y, text_y], color='grey', linewidth=1)
            
            # Aggiungi il numero della curva nel cerchio
            ax.text(text_x, text_y, txt, va='center_baseline', ha='center', 
                   size='small', color='white', weight='bold', zorder=11)
        
        # Configura gli assi
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Calcola i limiti con margine appropriato
        all_x = np.concatenate([rotated_track[:, 0], 
                               [rotate([corner['X'] + 500, corner['Y']], angle=track_angle)[0] 
                                for _, corner in circuit_info.corners.iterrows()]])
        all_y = np.concatenate([rotated_track[:, 1], 
                               [rotate([corner['X'], corner['Y'] + 500], angle=track_angle)[1] 
                                for _, corner in circuit_info.corners.iterrows()]])
        
        margin_x = (max(all_x) - min(all_x)) * 0.1
        margin_y = (max(all_y) - min(all_y)) * 0.1
        
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
        
        # Salva l'immagine
        img = io.BytesIO()
        plt.savefig(img, format='png', 
                   bbox_inches='tight', 
                   dpi=150,
                   transparent=True, 
                   facecolor='none', 
                   edgecolor='none',
                   pad_inches=0.3)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return plot_url
    except Exception as e:
        print(f"Errore nel generare il grafico del circuito: {e}")
        return None

def get_weather_info(session):
    """Ottiene informazioni meteo"""
    try:
        weather = session.weather_data
        if not weather.empty:
            latest_weather = weather.iloc[-1]
            return {
                'temperature': f"{latest_weather.get('AirTemp', 'N/A')}°C",
                'humidity': f"{latest_weather.get('Humidity', 'N/A')}%",
                'pressure': f"{latest_weather.get('Pressure', 'N/A')} mbar",
                'wind_speed': f"{latest_weather.get('WindSpeed', 'N/A')} km/h"
            }
    except Exception as e:
        print(f"Errore nel recuperare dati meteo: {e}")
    
    return {
        'temperature': 'N/A',
        'humidity': 'N/A', 
        'pressure': 'N/A',
        'wind_speed': 'N/A'
    }

def create_speed_plot(driver_data):
    """Crea grafico interattivo della velocità con palette F1"""
    try:
        # Usa direttamente driver_data invece di driver_data.laps
        if driver_data.empty:
            return None
        
        # Prendi solo i giri validi (non pit stop o out laps)
        valid_laps = driver_data[
            (driver_data['LapTime'].notna()) & 
            (driver_data['LapTime'] > pd.Timedelta(seconds=30))  # Filtra giri troppo veloci (probabilmente errori)
        ]
        
        if valid_laps.empty:
            return None
            
        lap_numbers = valid_laps['LapNumber']
        speeds = []
        
        for _, lap in valid_laps.iterrows():
            try:
                telemetry = lap.get_telemetry()
                if not telemetry.empty and 'Speed' in telemetry.columns:
                    avg_speed = telemetry['Speed'].mean()
                    if not pd.isna(avg_speed) and avg_speed > 0:
                        speeds.append(avg_speed)
                    else:
                        speeds.append(None)
                else:
                    speeds.append(None)
            except Exception as e:
                print(f"Errore nel recuperare telemetria per giro {lap.get('LapNumber', 'N/A')}: {e}")
                speeds.append(None)
        
        # Rimuovi i valori None
        valid_data = [(ln, s) for ln, s in zip(lap_numbers, speeds) if s is not None]
        if not valid_data:
            return None
            
        lap_nums, speed_vals = zip(*valid_data)
        
        fig = px.line(x=lap_nums, y=speed_vals, 
                     title='Velocità Media per Giro',
                     labels={'x': 'Numero Giro', 'y': 'Velocità (km/h)'})
        
        # Palette colori F1
        fig.update_traces(line_color='#E10600', line_width=4)  # Rosso F1
        fig.update_layout(
            font_family="Orbitron",
            plot_bgcolor='rgba(0,0,0,0)',  # sfondo trasparente
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            title_font_color='#E10600',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )

        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Errore nel creare grafico velocità: {e}")
        return None

def create_tire_plot(driver_data):
    """Crea grafico interattivo dell'usura gomme con colori F1 - FORMATO TEMPO CORRETTO"""
    try:
        # Usa direttamente driver_data
        if driver_data.empty:
            return None
        
        # Filtra giri validi
        valid_laps = driver_data[
            (driver_data['LapTime'].notna()) & 
            (driver_data['Compound'].notna()) &
            (driver_data['LapTime'] > pd.Timedelta(seconds=30))
        ]
        
        if valid_laps.empty:
            return None
        
        # Colori gomme F1 ufficiali
        tire_colors = {
            'SOFT': '#E10600',      # Rosso
            'MEDIUM': '#FFD700',    # Giallo 
            'HARD': '#FFFFFF',      # Bianco
            'INTERMEDIATE': '#39FF14', # Verde
            'WET': '#0080FF'        # Blu
        }
        
        fig = go.Figure()
        
        for compound in valid_laps['Compound'].unique():
            compound_laps = valid_laps[valid_laps['Compound'] == compound]
            
            # Converti i tempi giro in formato mm:ss:mmm per il display
            lap_times_formatted = [format_laptime(lt) for lt in compound_laps['LapTime']]
            # Mantieni i secondi per il grafico
            lap_times_seconds = [lt.total_seconds() if pd.notna(lt) else 0 for lt in compound_laps['LapTime']]
            
            fig.add_trace(go.Scatter(
                x=compound_laps['LapNumber'],
                y=lap_times_seconds,
                mode='markers+lines',
                name=f'Gomme {compound}',
                marker_color=tire_colors.get(compound, '#CCCCCC'),
                line=dict(width=4),
                text=lap_times_formatted,  # Mostra il formato mm:ss:mmm nell'hover
                hovertemplate='<b>Giro %{x}</b><br>Tempo: %{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Prestazioni per Tipo di Gomma',
            xaxis_title='Numero Giro',
            yaxis_title='Tempo Giro (secondi)',
            font_family="Orbitron",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            title_font_color='#E10600',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )

        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Errore nel creare grafico gomme: {e}")
        return None

def create_position_plot(session, driver):
    """Crea grafico delle posizioni durante la gara con stile F1 - VERSIONE CORRETTA"""
    try:
        # Usa il metodo corretto per ottenere i dati dei piloti
        driver_laps = session.laps.pick_drivers(driver)
        
        if driver_laps.empty:
            return None
        
        # Crea il grafico con Plotly per coerenza con gli altri grafici
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['Position'],
            mode='markers+lines',
            name=driver,
            line=dict(color='#E10600', width=4),
            marker=dict(size=6, color='#E10600')
        ))
        
        fig.update_layout(
            title='Andamento Posizione Durante la Gara',
            xaxis_title='Numero Giro',
            yaxis_title='Posizione',
            font_family="Orbitron",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            title_font_color='#E10600',
            xaxis=dict(showgrid=False),
            yaxis=dict(
                autorange="reversed",
                tick0=1,
                dtick=1,
                showgrid=False  # rimuove la griglia
            )
        )

        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Errore nel creare grafico posizioni: {e}")
        return None

def get_fastest_lap_info(driver_data):
    """Ottiene informazioni sul giro più veloce - FORMATO TEMPO CORRETTO"""
    try:
        fastest_lap = driver_data.pick_fastest()
        if fastest_lap is None:
            return None
            
        telemetry = fastest_lap.get_telemetry()
        
        # Formatta il tempo nel formato mm:ss:mmm
        formatted_time = format_laptime(fastest_lap['LapTime'])
        
        return {
            'lap_time': formatted_time,
            'lap_number': int(fastest_lap['LapNumber']),
            'avg_speed': f"{telemetry['Speed'].mean():.1f} km/h" if not telemetry.empty else "N/A"
        }
    except Exception as e:
        print(f"Errore nel calcolare giro più veloce: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', 
                         piloti_per_anno=piloti_per_anno,
                         gp_per_anno=gp_per_anno)

@app.route('/get_pilots')
def get_pilots():
    year = int(request.args.get('year'))
    return jsonify(piloti_per_anno.get(year, []))

@app.route('/get_gps')
def get_gps():
    year = int(request.args.get('year'))
    return jsonify(gp_per_anno.get(year, []))

@app.route('/analyze')
def analyze():
    year = int(request.args.get('year'))
    gp = request.args.get('gp')
    driver = request.args.get('driver')
    
    try:
        # Carica sessione gara
        session = fastf1.get_session(year, gp, 'R')
        session.load()
        
        # Ottieni dati pilota
        driver_data = session.laps[session.laps['Driver'] == driver]
        
        # Genera grafici e dati con gestione errori migliorata
        circuit_plot = get_circuit_plot(session)
        weather_info = get_weather_info(session)
        
        # Passa direttamente driver_data invece di richiamare session.laps
        speed_plot = create_speed_plot(driver_data)
        tire_plot = create_tire_plot(driver_data)
        position_plot = create_position_plot(session, driver)
        fastest_lap_info = get_fastest_lap_info(driver_data)
        
        # Simulazione analisi rete neurale (da implementare)
        neural_analysis = {
            'optimal_pitstop': f"Giro {np.random.randint(15, 25)}",
            'actual_pitstop': f"Giro {np.random.randint(18, 28)}",
            'position_prediction': f"{np.random.randint(3, 8)}ª posizione",
            'actual_position': f"{np.random.randint(2, 10)}ª posizione",
            'analysis': "L'analisi della rete neurale suggerisce che il pit stop ottimale sarebbe stato anticipato di 3-5 giri per massimizzare le prestazioni in base alle condizioni della pista e al degrado delle gomme."
        }
        
        return render_template('results.html',
                             year=year,
                             gp=gp,
                             driver=driver,
                             circuit_plot=circuit_plot,
                             weather_info=weather_info,
                             speed_plot=speed_plot,
                             tire_plot=tire_plot,
                             position_plot=position_plot,
                             fastest_lap_info=fastest_lap_info,
                             neural_analysis=neural_analysis)
                             
    except Exception as e:
        print(f"Errore nell'analisi: {e}")
        return f"Errore nell'analisi dei dati: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)