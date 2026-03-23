"""
Análisis de Carga Eléctrica - Aplicación Streamlit
Sistema completo con gráficos, calculadora de proyección y configuración de intervalos
Organizado en pestañas para mejor navegación
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Carga Eléctrica",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    div[data-testid="metric-container"] { background: white; border-radius: 10px; padding: 15px; 
                                          box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; }
    .calc-result { background: #f0f7ff; border-left: 4px solid #1E88E5; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .sidebar-config { background: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    div[data-baseweb="tab"] { padding: 10px 20px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ==================== FUNCIONES DE PARSEO ====================

def find_column(df, possible_names):
    df_cols = [str(c).strip().upper() for c in df.columns]
    for name in possible_names:
        name_upper = str(name).strip().upper()
        for i, col in enumerate(df_cols):
            if name_upper == col or name_upper in col or col in name_upper:
                return df.columns[i]
    return None


def find_header_row(df, max_rows=10):
    header_keywords = ['STARTTIME', 'U1AVG', 'STOTAVG', 'STOT', 'WTOTAVG']
    best_row, best_score = 0, 0
    for idx in range(min(max_rows, len(df))):
        row_str = ' '.join([str(v).upper() for v in df.iloc[idx].values if pd.notna(v)])
        score = sum(1 for kw in header_keywords if kw in row_str)
        if score > best_score:
            best_score, best_row = score, idx
        if score >= 3:
            return idx
    return best_row


def parse_transformer_info(df_diagramas):
    info = {'sed': 'N/A', 'kva_nominal': None, 'tipo': 'N/A', 'tension': 'N/A'}
    
    for idx, row in df_diagramas.iterrows():
        for col_idx in range(len(row.values) - 1):
            cell = row.values[col_idx]
            next_cell = row.values[col_idx + 1] if col_idx + 1 < len(row.values) else None
            
            if pd.isna(cell):
                continue
            cell_str = str(cell).upper().strip()
            
            if cell_str == 'KVA NOMINAL':
                if pd.notna(next_cell):
                    try:
                        num_val = float(next_cell)
                        if 10 < num_val < 5000:
                            info['kva_nominal'] = num_val
                    except:
                        pass
            elif cell_str == 'SED':
                if pd.notna(next_cell):
                    info['sed'] = str(next_cell)
    
    if info['kva_nominal'] is None:
        info['kva_nominal'] = 100
    
    return info


def parse_measurements(df_data):
    header_row = find_header_row(df_data)
    
    if header_row >= 0:
        headers = [str(h).strip() if pd.notna(h) else f'col_{i}' for i, h in enumerate(df_data.iloc[header_row].values)]
        df_result = df_data.iloc[header_row + 1:].copy()
        df_result.columns = headers
        df_result = df_result.reset_index(drop=True)
    else:
        df_result = df_data.copy()
    
    column_mapping = {
        'starttime': ['starttime', 'start_time', 'fecha', 'time', 'timestamp'],
        'U1Avg': ['u1avg', 'u1'], 'U2Avg': ['u2avg', 'u2'], 'U3Avg': ['u3avg', 'u3'],
        'I1Avg': ['i1avg', 'i1'], 'I2Avg': ['i2avg', 'i2'], 'I3Avg': ['i3avg', 'i3'],
        'STotAvg': ['stotavg', 'stot', 's_tot', 'stotal'],
        'WTotAvg': ['wtotavg', 'wtot', 'w_tot', 'wtotal'],
        'PST1': ['pst1'], 'PST2': ['pst2'], 'PST3': ['pst3'],
        'TDD_U1': ['tdd_u1', 'tdd1'], 'TDD_U2': ['tdd_u2', 'tdd2'], 'TDD_U3': ['tdd_u3', 'tdd3'],
    }
    
    final_df = pd.DataFrame()
    for standard_name, possible_names in column_mapping.items():
        found_col = find_column(df_result, possible_names)
        if found_col is not None:
            final_df[standard_name] = df_result[found_col]
        elif standard_name in ['PST1', 'PST2', 'PST3', 'TDD_U1', 'TDD_U2', 'TDD_U3']:
            final_df[standard_name] = 0.0
    
    numeric_cols = ['U1Avg', 'U2Avg', 'U3Avg', 'I1Avg', 'I2Avg', 'I3Avg', 
                    'STotAvg', 'WTotAvg', 'PST1', 'PST2', 'PST3', 'TDD_U1', 'TDD_U2', 'TDD_U3']
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
    
    if 'starttime' in final_df.columns:
        final_df['starttime'] = pd.to_datetime(final_df['starttime'], errors='coerce')
    
    if 'WTotAvg' not in final_df.columns or final_df['WTotAvg'].sum() == 0:
        if 'STotAvg' in final_df.columns:
            final_df['WTotAvg'] = final_df['STotAvg'] * 0.95
    
    if 'STotAvg' in final_df.columns:
        final_df = final_df.dropna(subset=['STotAvg'])
        final_df = final_df[final_df['STotAvg'] > 0]
    
    return final_df


def resample_data(df, interval_minutes):
    """Remuestrea los datos al intervalo especificado"""
    if interval_minutes <= 0 or 'starttime' not in df.columns:
        return df
    
    df_resampled = df.copy()
    df_resampled = df_resampled.set_index('starttime')
    
    # Resamplear y promediar
    agg_dict = {
        'U1Avg': 'mean', 'U2Avg': 'mean', 'U3Avg': 'mean',
        'I1Avg': 'mean', 'I2Avg': 'mean', 'I3Avg': 'mean',
        'STotAvg': 'mean', 'WTotAvg': 'mean',
        'PST1': 'mean', 'PST2': 'mean', 'PST3': 'mean',
        'TDD_U1': 'mean', 'TDD_U2': 'mean', 'TDD_U3': 'mean'
    }
    
    # Solo incluir columnas que existen
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_resampled.columns}
    
    df_resampled = df_resampled.resample(f'{interval_minutes}T').agg(agg_dict)
    df_resampled = df_resampled.dropna(subset=['STotAvg'])
    df_resampled = df_resampled.reset_index()
    
    return df_resampled


def calculate_kpis(df, kva_nominal):
    kpis = {}
    kpis['max_power_kva'] = df['STotAvg'].max() / 1000
    kpis['min_power_kva'] = df['STotAvg'].min() / 1000
    kpis['avg_power_kva'] = df['STotAvg'].mean() / 1000
    kpis['utilization'] = (df['STotAvg'].max() / (kva_nominal * 1000)) * 100
    
    if 'WTotAvg' in df.columns and df['WTotAvg'].sum() > 0:
        valid_pf = df[df['STotAvg'] > 0]
        kpis['power_factor'] = min(max((valid_pf['WTotAvg'] / valid_pf['STotAvg']).mean(), 0), 1)
    else:
        kpis['power_factor'] = 0.95
    
    tdd_cols = [col for col in ['TDD_U1', 'TDD_U2', 'TDD_U3'] if col in df.columns]
    kpis['max_tdd'] = df[tdd_cols].max().max() if tdd_cols else 0
    
    pst_cols = [col for col in ['PST1', 'PST2', 'PST3'] if col in df.columns]
    kpis['max_pst'] = df[pst_cols].max().max() if pst_cols else 0
    
    u_cols = [col for col in ['U1Avg', 'U2Avg', 'U3Avg'] if col in df.columns]
    if len(u_cols) >= 2:
        u_max, u_min, u_avg = df[u_cols].max(axis=1), df[u_cols].min(axis=1), df[u_cols].mean(axis=1)
        kpis['voltage_unbalance'] = ((u_max - u_min) / u_avg * 100).max()
    else:
        kpis['voltage_unbalance'] = 0
    
    i_cols = [col for col in ['I1Avg', 'I2Avg', 'I3Avg'] if col in df.columns]
    if len(i_cols) >= 2:
        i_max, i_min, i_avg = df[i_cols].max(axis=1), df[i_cols].min(axis=1), df[i_cols].mean(axis=1)
        kpis['current_unbalance'] = ((i_max - i_min) / i_avg * 100).max()
    else:
        kpis['current_unbalance'] = 0
    
    if u_cols:
        kpis['avg_voltage'] = df[u_cols].mean().mean()
        kpis['min_voltage'] = df[u_cols].min().min()
        kpis['max_voltage'] = df[u_cols].max().max()
    else:
        kpis['avg_voltage'] = kpis['min_voltage'] = kpis['max_voltage'] = 230
    
    if i_cols:
        kpis['avg_current'] = df[i_cols].mean().mean()
        kpis['max_current'] = df[i_cols].max().max()
    else:
        kpis['avg_current'] = kpis['max_current'] = 0
    
    kpis['num_measurements'] = len(df)
    
    if 'starttime' in df.columns and df['starttime'].notna().any():
        kpis['max_datetime'] = df.loc[df['STotAvg'].idxmax(), 'starttime']
        kpis['min_datetime'] = df.loc[df['STotAvg'].idxmin(), 'starttime']
        kpis['max_date'] = kpis['max_datetime'].date()
        kpis['min_date'] = kpis['min_datetime'].date()
        kpis['max_hour'] = kpis['max_datetime'].hour + kpis['max_datetime'].minute / 60
        kpis['min_hour'] = kpis['min_datetime'].hour + kpis['min_datetime'].minute / 60
    
    return kpis


# ==================== FUNCIONES DE PERFILES ====================

def calculate_daily_profile(df, interval_minutes):
    """Calcula el perfil diario promedio (24 horas) con el intervalo especificado"""
    df_temp = df.copy()
    df_temp['hora'] = df_temp['starttime'].dt.hour
    df_temp['minuto'] = df_temp['starttime'].dt.minute
    
    # Crear grupos por hora y minutos redondeados al intervalo
    df_temp['intervalo'] = (df_temp['minuto'] // interval_minutes) * interval_minutes
    df_temp['hora_decimal'] = df_temp['hora'] + df_temp['intervalo'] / 60
    
    profile = df_temp.groupby(['hora', 'intervalo'])['STotAvg'].agg(['mean', 'max', 'min', 'std']).reset_index()
    profile.columns = ['hora', 'intervalo', 'promedio', 'maximo', 'minimo', 'desv_std']
    profile['promedio_kva'] = profile['promedio'] / 1000
    profile['maximo_kva'] = profile['maximo'] / 1000
    profile['minimo_kva'] = profile['minimo'] / 1000
    profile['hora_decimal'] = profile['hora'] + profile['intervalo'] / 60
    
    return profile.sort_values('hora_decimal')


def calculate_weekly_profile(df, interval_minutes):
    """Calcula el perfil semanal continuo (7 días × 24 horas)"""
    df_temp = df.copy()
    df_temp['dia_semana'] = df_temp['starttime'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df_temp['hora'] = df_temp['starttime'].dt.hour
    df_temp['minuto'] = df_temp['starttime'].dt.minute
    df_temp['intervalo'] = (df_temp['minuto'] // interval_minutes) * interval_minutes
    
    dias_nombres = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    
    profile = df_temp.groupby(['dia_semana', 'hora', 'intervalo'])['STotAvg'].agg(['mean', 'max', 'min']).reset_index()
    profile.columns = ['dia_semana', 'hora', 'intervalo', 'promedio', 'maximo', 'minimo']
    profile['promedio_kva'] = profile['promedio'] / 1000
    profile['maximo_kva'] = profile['maximo'] / 1000
    profile['minimo_kva'] = profile['minimo'] / 1000
    profile['dia_nombre'] = profile['dia_semana'].map(dias_nombres)
    
    # Crear hora continua para gráfico (0 = lunes 00:00, 168 = domingo 24:00)
    profile['hora_continua'] = profile['dia_semana'] * 24 + profile['hora'] + profile['intervalo'] / 60
    
    return profile.sort_values('hora_continua'), dias_nombres


def calculate_total_profile(df):
    """Prepara datos para el gráfico total"""
    df_temp = df.copy()
    df_temp['fecha'] = df_temp['starttime'].dt.date
    df_temp['hora_decimal'] = df_temp['starttime'].dt.hour + df_temp['starttime'].dt.minute / 60
    df_temp['potencia_kva'] = df_temp['STotAvg'] / 1000
    return df_temp


# ==================== FUNCIONES DE CÁLCULO ====================

def calculate_daily_projection(daily_profile, measured_value, measured_hour, interval_minutes):
    """Calcula la proyección basada en el perfil diario"""
    profile = daily_profile.copy()
    
    # Encontrar el intervalo más cercano
    profile['diff'] = (profile['hora_decimal'] - measured_hour).abs()
    idx_measured = profile['diff'].idxmin()
    profile_value = profile.loc[idx_measured, 'promedio_kva']
    
    if profile_value > 0:
        factor = measured_value / profile_value
    else:
        factor = 1
    
    profile['proyeccion'] = profile['promedio_kva'] * factor
    
    max_idx = profile['proyeccion'].idxmax()
    min_idx = profile['proyeccion'].idxmin()
    
    result = {
        'factor_proporcion': factor,
        'valor_max_proyectado': profile.loc[max_idx, 'proyeccion'],
        'hora_max_proyectado': profile.loc[max_idx, 'hora_decimal'],
        'valor_min_proyectado': profile.loc[min_idx, 'proyeccion'],
        'hora_min_proyectado': profile.loc[min_idx, 'hora_decimal'],
        'intervalo_minutos': interval_minutes,
    }
    
    return result, profile


def calculate_weekly_projection(weekly_profile, measured_value, measured_day, measured_hour, dias_nombres, interval_minutes):
    """Calcula la proyección basada en el perfil semanal"""
    profile = weekly_profile.copy()
    
    dias_numeros = {v: k for k, v in dias_nombres.items()}
    measured_day_num = dias_numeros.get(measured_day, 0)
    
    # Encontrar punto más cercano
    mask = (profile['dia_semana'] == measured_day_num)
    day_profile = profile[mask].copy()
    day_profile['diff'] = (day_profile['hora'] + day_profile['intervalo']/60 - measured_hour).abs()
    
    if len(day_profile) > 0:
        idx_measured = day_profile['diff'].idxmin()
        profile_value = profile.loc[idx_measured, 'promedio_kva']
    else:
        profile_value = profile['promedio_kva'].mean()
    
    if profile_value > 0:
        factor = measured_value / profile_value
    else:
        factor = 1
    
    profile['proyeccion'] = profile['promedio_kva'] * factor
    
    max_idx = profile['proyeccion'].idxmax()
    min_idx = profile['proyeccion'].idxmin()
    
    result = {
        'factor_proporcion': factor,
        'valor_max_proyectado': profile.loc[max_idx, 'proyeccion'],
        'dia_max': profile.loc[max_idx, 'dia_nombre'],
        'hora_max': profile.loc[max_idx, 'hora'],
        'intervalo_max': profile.loc[max_idx, 'intervalo'],
        'hora_continua_max': profile.loc[max_idx, 'hora_continua'],
        'valor_min_proyectado': profile.loc[min_idx, 'proyeccion'],
        'dia_min': profile.loc[min_idx, 'dia_nombre'],
        'hora_min': profile.loc[min_idx, 'hora'],
        'intervalo_min': profile.loc[min_idx, 'intervalo'],
        'hora_continua_min': profile.loc[min_idx, 'hora_continua'],
    }
    
    return result, profile


def calculate_total_projection(df, measured_value, measured_date, measured_hour, use_max_day=False, use_min_day=False):
    """Calcula la proyección basada en un día específico"""
    df_temp = df.copy()
    df_temp['fecha'] = df_temp['starttime'].dt.date
    df_temp['potencia_kva'] = df_temp['STotAvg'] / 1000
    
    if use_max_day:
        max_date = df_temp.loc[df_temp['STotAvg'].idxmax(), 'starttime'].date()
        target_data = df_temp[df_temp['fecha'] == max_date].copy()
        base_label = f"día de máxima carga ({max_date})"
    elif use_min_day:
        min_date = df_temp.loc[df_temp['STotAvg'].idxmin(), 'starttime'].date()
        target_data = df_temp[df_temp['fecha'] == min_date].copy()
        base_label = f"día de mínima carga ({min_date})"
    else:
        target_data = df_temp[df_temp['fecha'] == measured_date].copy()
        base_label = f"día {measured_date}"
    
    if len(target_data) == 0:
        return None, None, "No hay datos para la fecha especificada"
    
    target_data['hora_decimal'] = target_data['starttime'].dt.hour + target_data['starttime'].dt.minute / 60
    
    idx_measured = (target_data['hora_decimal'] - measured_hour).abs().idxmin()
    profile_value = target_data.loc[idx_measured, 'potencia_kva']
    
    if profile_value > 0:
        factor = measured_value / profile_value
    else:
        factor = 1
    
    target_data['proyeccion'] = target_data['potencia_kva'] * factor
    
    max_idx = target_data['proyeccion'].idxmax()
    min_idx = target_data['proyeccion'].idxmin()
    
    result = {
        'factor_proporcion': factor,
        'base_calculo': base_label,
        'valor_max_proyectado': target_data.loc[max_idx, 'proyeccion'],
        'hora_max_proyectado': target_data.loc[max_idx, 'hora_decimal'],
        'datetime_max': target_data.loc[max_idx, 'starttime'],
        'valor_min_proyectado': target_data.loc[min_idx, 'proyeccion'],
        'hora_min_proyectado': target_data.loc[min_idx, 'hora_decimal'],
        'datetime_min': target_data.loc[min_idx, 'starttime'],
        'medida_original': measured_value,
        'hora_medida': measured_hour,
    }
    
    return result, target_data, None


# ==================== FUNCIONES DE GRÁFICOS ====================

def add_max_min_markers(fig, x_data, y_data, x_max, y_max, x_min, y_min, show_max=True, show_min=True):
    """Agrega marcadores de máximo y mínimo a un gráfico"""
    
    if show_max and y_max is not None:
        # Marcador de máximo con estrella y flecha
        fig.add_trace(go.Scatter(
            x=[x_max], y=[y_max],
            mode='markers+text',
            marker=dict(symbol='star', size=15, color='red'),
            text=['⬆ MÁX'],
            textposition='top center',
            textfont=dict(size=12, color='red'),
            name='Máximo',
            hovertemplate=f'Máximo: {y_max:.2f} kVA<extra></extra>'
        ))
    
    if show_min and y_min is not None:
        # Marcador de mínimo con estrella y flecha
        fig.add_trace(go.Scatter(
            x=[x_min], y=[y_min],
            mode='markers+text',
            marker=dict(symbol='star', size=15, color='green'),
            text=['⬇ MÍN'],
            textposition='bottom center',
            textfont=dict(size=12, color='green'),
            name='Mínimo',
            hovertemplate=f'Mínimo: {y_min:.2f} kVA<extra></extra>'
        ))
    
    return fig


def plot_daily_profile(daily_profile, projection_profile=None, show_markers=True):
    """Gráfico del perfil diario promedio con marcadores"""
    fig = go.Figure()
    
    # Área del promedio
    fig.add_trace(go.Scatter(
        x=daily_profile['hora_decimal'], y=daily_profile['promedio_kva'],
        name='Promedio', line=dict(color='#1E88E5', width=3),
        fill='tozeroy', fillcolor='rgba(30, 136, 229, 0.2)'
    ))
    
    # Línea de máximo
    fig.add_trace(go.Scatter(
        x=daily_profile['hora_decimal'], y=daily_profile['maximo_kva'],
        name='Máximo histórico', line=dict(color='#E53935', width=1.5, dash='dot')
    ))
    
    # Línea de mínimo
    fig.add_trace(go.Scatter(
        x=daily_profile['hora_decimal'], y=daily_profile['minimo_kva'],
        name='Mínimo histórico', line=dict(color='#43A047', width=1.5, dash='dot')
    ))
    
    # Proyección si existe
    if projection_profile is not None:
        fig.add_trace(go.Scatter(
            x=projection_profile['hora_decimal'], y=projection_profile['proyeccion'],
            name='Proyección', line=dict(color='#FF9800', width=3, dash='dash')
        ))
    
    # Encontrar máximos y mínimos del promedio
    max_idx = daily_profile['promedio_kva'].idxmax()
    min_idx = daily_profile['promedio_kva'].idxmin()
    x_max = daily_profile.loc[max_idx, 'hora_decimal']
    y_max = daily_profile.loc[max_idx, 'promedio_kva']
    x_min = daily_profile.loc[min_idx, 'hora_decimal']
    y_min = daily_profile.loc[min_idx, 'promedio_kva']
    
    # Agregar marcadores
    if show_markers:
        fig = add_max_min_markers(fig, daily_profile['hora_decimal'], daily_profile['promedio_kva'],
                                   x_max, y_max, x_min, y_min)
    
    fig.update_layout(
        title='📊 Perfil Diario Promedio',
        xaxis_title='Hora del día', yaxis_title='Potencia (kVA)',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0, 24]),
        hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def plot_weekly_profile(weekly_profile, projection_profile=None, show_markers=True):
    """Gráfico del perfil semanal CONTINUO con marcadores"""
    fig = go.Figure()
    
    # Línea continua del promedio semanal
    fig.add_trace(go.Scatter(
        x=weekly_profile['hora_continua'], y=weekly_profile['promedio_kva'],
        name='Promedio', line=dict(color='#1E88E5', width=2.5),
        mode='lines'
    ))
    
    # Área sombreada
    fig.add_trace(go.Scatter(
        x=weekly_profile['hora_continua'], y=weekly_profile['maximo_kva'],
        name='Máximo', line=dict(color='#E53935', width=1, dash='dot'),
        fill=None, showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=weekly_profile['hora_continua'], y=weekly_profile['minimo_kva'],
        name='Mínimo', line=dict(color='#43A047', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(100, 100, 100, 0.1)'
    ))
    
    # Proyección si existe
    if projection_profile is not None:
        fig.add_trace(go.Scatter(
            x=projection_profile['hora_continua'], y=projection_profile['proyeccion'],
            name='Proyección', line=dict(color='#FF9800', width=2.5, dash='dash')
        ))
    
    # Encontrar máximos y mínimos
    max_idx = weekly_profile['promedio_kva'].idxmax()
    min_idx = weekly_profile['promedio_kva'].idxmin()
    x_max = weekly_profile.loc[max_idx, 'hora_continua']
    y_max = weekly_profile.loc[max_idx, 'promedio_kva']
    x_min = weekly_profile.loc[min_idx, 'hora_continua']
    y_min = weekly_profile.loc[min_idx, 'promedio_kva']
    
    # Agregar marcadores
    if show_markers:
        fig = add_max_min_markers(fig, weekly_profile['hora_continua'], weekly_profile['promedio_kva'],
                                   x_max, y_max, x_min, y_min)
    
    # Líneas verticales para separar días
    for i in range(1, 7):
        fig.add_vline(x=i*24, line_dash='dash', line_color='gray', line_width=0.5, opacity=0.5)
    
    # Etiquetas de días en el eje X
    dias_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    tickvals = [12 + i*24 for i in range(7)]
    
    fig.update_layout(
        title='📊 Perfil Semanal Continuo (Lunes a Domingo)',
        xaxis_title='Día de la semana', yaxis_title='Potencia (kVA)',
        xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=dias_labels),
        hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def plot_total_profile(df, highlight_date=None, projection_data=None, show_markers=True):
    """Gráfico total con todos los días y marcadores"""
    df_temp = df.copy()
    df_temp['fecha_str'] = df_temp['starttime'].dt.strftime('%Y-%m-%d')
    df_temp['potencia_kva'] = df_temp['STotAvg'] / 1000
    
    fig = go.Figure()
    
    # Todos los días en gris claro
    for fecha in df_temp['fecha_str'].unique():
        data = df_temp[df_temp['fecha_str'] == fecha]
        fig.add_trace(go.Scatter(
            x=data['starttime'], y=data['potencia_kva'],
            mode='lines', line=dict(color='lightgray', width=0.5),
            showlegend=False, hoverinfo='skip'
        ))
    
    # Promedio general
    df_hourly = df_temp.groupby(df_temp['starttime'].dt.floor('H'))['potencia_kva'].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=df_hourly['starttime'], y=df_hourly['potencia_kva'],
        name='Promedio', line=dict(color='#1E88E5', width=2)
    ))
    
    # Día destacado
    if highlight_date is not None:
        highlight_data = df_temp[df_temp['starttime'].dt.date == highlight_date]
        if len(highlight_data) > 0:
            fig.add_trace(go.Scatter(
                x=highlight_data['starttime'], y=highlight_data['potencia_kva'],
                name='Día seleccionado', line=dict(color='#E53935', width=2.5)
            ))
    
    # Proyección
    if projection_data is not None:
        fig.add_trace(go.Scatter(
            x=projection_data['starttime'], y=projection_data['proyeccion'],
            name='Proyección', line=dict(color='#FF9800', width=2, dash='dash')
        ))
    
    # Encontrar máximos y mínimos globales
    max_idx = df_temp['potencia_kva'].idxmax()
    min_idx = df_temp['potencia_kva'].idxmin()
    x_max = df_temp.loc[max_idx, 'starttime']
    y_max = df_temp.loc[max_idx, 'potencia_kva']
    x_min = df_temp.loc[min_idx, 'starttime']
    y_min = df_temp.loc[min_idx, 'potencia_kva']
    
    # Agregar marcadores
    if show_markers:
        fig = add_max_min_markers(fig, df_temp['starttime'], df_temp['potencia_kva'],
                                   x_max, y_max, x_min, y_min)
    
    fig.update_layout(
        title='📊 Perfil Total - Todos los Días',
        xaxis_title='Fecha/Hora', yaxis_title='Potencia (kVA)',
        hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def plot_voltage_chart(df, show_markers=True):
    fig = go.Figure()
    colors = ['#E53935', '#FB8C00', '#43A047']
    phases = ['U1Avg', 'U2Avg', 'U3Avg']
    names = ['Fase 1', 'Fase 2', 'Fase 3']
    
    for i, (phase, name) in enumerate(zip(phases, names)):
        if phase in df.columns:
            fig.add_trace(go.Scatter(
                x=df['starttime'], y=df[phase], name=name,
                line=dict(color=colors[i], width=1.5), opacity=0.8
            ))
    
    fig.add_hline(y=253, line_dash='dash', line_color='red', annotation_text='+10%')
    fig.add_hline(y=207, line_dash='dash', line_color='red', annotation_text='-10%')
    
    fig.update_layout(
        title='Voltaje por Fase', xaxis_title='Fecha/Hora', yaxis_title='Voltaje (V)',
        hovermode='x unified', height=350
    )
    return fig


def plot_current_chart(df, show_markers=True):
    fig = go.Figure()
    colors = ['#E53935', '#FB8C00', '#43A047']
    phases = ['I1Avg', 'I2Avg', 'I3Avg']
    names = ['Fase 1', 'Fase 2', 'Fase 3']
    
    for i, (phase, name) in enumerate(zip(phases, names)):
        if phase in df.columns:
            fig.add_trace(go.Scatter(
                x=df['starttime'], y=df[phase], name=name,
                line=dict(color=colors[i], width=1.5), opacity=0.8
            ))
    
    fig.update_layout(
        title='Corriente por Fase', xaxis_title='Fecha/Hora', yaxis_title='Corriente (A)',
        hovermode='x unified', height=350
    )
    return fig


# ==================== APLICACIÓN PRINCIPAL ====================

def main():
    st.markdown('<p class="main-header">⚡ Análisis de Carga Eléctrica</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema completo con gráficos y calculadora de proyección</p>', unsafe_allow_html=True)
    
    # ==================== SIDEBAR - CONFIGURACIÓN ====================
    st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=80)
    st.sidebar.title("⚙️ Configuración")
    
    # Intervalo de tiempo - SOLO 10 MIN Y 1 HORA
    st.sidebar.markdown("### 📐 Intervalo de datos")
    interval_options = {
        "Original (10 min)": 0,
        "1 hora": 60
    }
    
    selected_interval_name = st.sidebar.selectbox(
        "Intervalo de visualización",
        list(interval_options.keys()),
        index=0,
        help="Selecciona el intervalo para agrupar los datos"
    )
    interval_minutes = interval_options[selected_interval_name]
    
    # Mostrar marcadores
    show_markers = st.sidebar.checkbox("Mostrar marcadores máx/mín", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.title("📁 Archivo")
    
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo (.xlsx)", type=['xlsx'])
    
    # Mostrar info del intervalo seleccionado
    if interval_minutes > 0:
        st.sidebar.info(f"📊 Datos agrupados cada {interval_minutes} minutos")
    else:
        st.sidebar.info("📊 Datos originales (10 min)")
    
    # ==================== PROCESAMIENTO ====================
    if uploaded_file is not None:
        try:
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None)
            
            diagramas_sheet = data_sheet = None
            data_sheet_name = ""
            
            for name, df in all_sheets.items():
                if 'DIAGRAMA' in name.upper():
                    diagramas_sheet = df
                else:
                    data_sheet = df
                    data_sheet_name = name
            
            if diagramas_sheet is None or data_sheet is None:
                st.error("❌ Faltan hojas en el archivo")
                return
            
            with st.spinner('Procesando...'):
                transformer_info = parse_transformer_info(diagramas_sheet)
                df_raw = parse_measurements(data_sheet)
                
                # Aplicar resampling según intervalo seleccionado
                df = resample_data(df_raw, interval_minutes)
                
                kpis = calculate_kpis(df, transformer_info['kva_nominal'])
                
                # Calcular perfiles con el intervalo
                daily_profile = calculate_daily_profile(df, interval_minutes if interval_minutes > 0 else 10)
                weekly_profile, dias_nombres = calculate_weekly_profile(df, interval_minutes if interval_minutes > 0 else 10)
                total_profile = calculate_total_profile(df)
            
            if len(df) == 0:
                st.error("❌ No hay datos válidos")
                return
            
            if transformer_info['sed'] == 'N/A':
                transformer_info['sed'] = data_sheet_name
            
            # === INFO TRANSFORMADOR ===
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("🏷️ SED", transformer_info['sed'])
            with col2: st.metric("⚡ Capacidad", f"{transformer_info['kva_nominal']} kVA")
            with col3: st.metric("📊 Registros", kpis['num_measurements'])
            with col4:
                if 'starttime' in df.columns:
                    st.metric("📅 Período", f"{(df['starttime'].max() - df['starttime'].min()).days} días")
            
            # Mostrar intervalo actual
            st.info(f"📐 **Intervalo seleccionado:** {selected_interval_name} | Datos procesados: {len(df)} registros")
            
            # === ESTADO ===
            critical = sum([kpis['utilization'] > 90, kpis['power_factor'] < 0.85, kpis['max_tdd'] > 8, 
                           kpis['max_pst'] > 1, kpis['voltage_unbalance'] > 5, kpis['current_unbalance'] > 15])
            warning = sum([75 < kpis['utilization'] <= 90, 0.85 <= kpis['power_factor'] < 0.92,
                          5 < kpis['max_tdd'] <= 8, 0.8 < kpis['max_pst'] <= 1])
            
            if critical > 0:
                st.error(f"🔴 Crítico: {critical} parámetros críticos")
            elif warning > 0:
                st.warning(f"🟡 Advertencia: {warning} parámetros")
            else:
                st.success("🟢 Estado Normal")
            
            # === KPIs EN FILA COMPACTA ===
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("⚡ Utilización", f"{kpis['utilization']:.1f}%")
            with col2: st.metric("🔋 Pot. Máx", f"{kpis['max_power_kva']:.2f} kVA", f"a las {kpis.get('max_hour', 0):.1f}h")
            with col3: st.metric("🌊 TDD Máx", f"{kpis['max_tdd']:.2f}%")
            with col4: st.metric("💡 PST Máx", f"{kpis['max_pst']:.3f}")
            
            # =====================================================
            # === PESTAÑAS PRINCIPALES ===
            # =====================================================
            st.markdown("---")
            main_tabs = st.tabs(["📊 Potencia", "📈 Voltaje/Corriente", "🧮 Calculadoras"])
            
            # ============== PESTAÑA: POTENCIA ==============
            with main_tabs[0]:
                st.markdown("### Análisis de Potencia")
                
                power_tabs = st.tabs(["📅 Diario", "📆 Semanal", "📊 Total"])
                
                # --- Diario ---
                with power_tabs[0]:
                    st.markdown(f"**Promedio de todos los días** | Intervalo: {selected_interval_name}")
                    st.plotly_chart(plot_daily_profile(daily_profile, show_markers=show_markers), use_container_width=True)
                
                # --- Semanal ---
                with power_tabs[1]:
                    st.markdown("**Lunes → Domingo (168 horas continuas)**")
                    st.plotly_chart(plot_weekly_profile(weekly_profile, show_markers=show_markers), use_container_width=True)
                
                # --- Total ---
                with power_tabs[2]:
                    col1, col2 = st.columns(2)
                    with col1:
                        available_dates = sorted(df['starttime'].dt.date.unique())
                        selected_date = st.selectbox("Resaltar día", 
                                                     ["Ninguno"] + [str(d) for d in available_dates],
                                                     key="total_date")
                        highlight_date = None if selected_date == "Ninguno" else datetime.strptime(selected_date, "%Y-%m-%d").date()
                    
                    with col2:
                        st.metric("🔺 Día MÁXIMA", str(kpis.get('max_date', 'N/A')))
                        st.metric("🔻 Día MÍNIMA", str(kpis.get('min_date', 'N/A')))
                    
                    st.plotly_chart(plot_total_profile(df, highlight_date, show_markers=show_markers), use_container_width=True)
            
            # ============== PESTAÑA: VOLTAJE/CORRIENTE ==============
            with main_tabs[1]:
                st.markdown("### Gráficos de Voltaje y Corriente")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_voltage_chart(df, show_markers=show_markers), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_current_chart(df, show_markers=show_markers), use_container_width=True)
                
                # Métricas adicionales
                st.markdown("#### Métricas de Calidad")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⚡ Voltaje Promedio", f"{kpis['avg_voltage']:.1f} V")
                    st.metric("🔺 Voltaje Máx", f"{kpis['max_voltage']:.1f} V")
                with col2:
                    st.metric("⚡ Voltaje Mín", f"{kpis['min_voltage']:.1f} V")
                    st.metric("🔺 Desbalance V", f"{kpis['voltage_unbalance']:.2f}%")
                with col3:
                    st.metric("🔌 Corriente Prom", f"{kpis['avg_current']:.1f} A")
                    st.metric("🔺 Corriente Máx", f"{kpis['max_current']:.1f} A")
                with col4:
                    st.metric("⚡ Factor de Potencia", f"{kpis['power_factor']:.3f}")
                    st.metric("🔺 Desbalance I", f"{kpis['current_unbalance']:.2f}%")
            
            # ============== PESTAÑA: CALCULADORAS ==============
            with main_tabs[2]:
                st.markdown("### Calculadoras de Proyección")
                
                calc_tabs = st.tabs(["📅 Diario", "📆 Semanal", "📊 Total"])
                
                # --- Calculadora Diaria ---
                with calc_tabs[0]:
                    st.markdown("#### Proyección basada en Perfil Diario")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        calc_power = st.number_input("Potencia medida (kVA)", min_value=0.0, value=100.0, step=1.0, key="daily_power")
                    with col2:
                        calc_hour = st.number_input("Hora de medición", min_value=0.0, max_value=23.99, value=12.0, step=0.25, key="daily_hour",
                                                   help="Hora en formato decimal (ej: 14.5 = 14:30)")
                    with col3:
                        st.metric("Intervalo actual", f"{interval_minutes if interval_minutes > 0 else 10} min")
                    
                    if st.button("🔮 Calcular Proyección Diaria", key="calc_daily", type="primary"):
                        result, proj_profile = calculate_daily_projection(daily_profile, calc_power, calc_hour, 
                                                                          interval_minutes if interval_minutes > 0 else 10)
                        
                        st.markdown("<div class='calc-result'>", unsafe_allow_html=True)
                        st.markdown(f"**📈 Factor de proporción:** `{result['factor_proporcion']:.4f}`")
                        st.markdown(f"**🔴 Valor MÁXIMO proyectado:** `{result['valor_max_proyectado']:.2f} kVA` a las **`{result['hora_max_proyectado']:.2f}h`**")
                        st.markdown(f"**🟢 Valor MÍNIMO proyectado:** `{result['valor_min_proyectado']:.2f} kVA` a las **`{result['hora_min_proyectado']:.2f}h`**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.plotly_chart(plot_daily_profile(daily_profile, proj_profile, show_markers=show_markers), use_container_width=True)
                
                # --- Calculadora Semanal ---
                with calc_tabs[1]:
                    st.markdown("#### Proyección basada en Perfil Semanal")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        calc_power_w = st.number_input("Potencia medida (kVA)", min_value=0.0, value=100.0, step=1.0, key="weekly_power")
                    with col2:
                        calc_day_w = st.selectbox("Día de medición", list(dias_nombres.values()), key="weekly_day")
                    with col3:
                        calc_hour_w = st.number_input("Hora", min_value=0.0, max_value=23.99, value=12.0, step=0.25, key="weekly_hour")
                    
                    if st.button("🔮 Calcular Proyección Semanal", key="calc_weekly", type="primary"):
                        result, proj_profile = calculate_weekly_projection(
                            weekly_profile, calc_power_w, calc_day_w, calc_hour_w, dias_nombres,
                            interval_minutes if interval_minutes > 0 else 10
                        )
                        
                        st.markdown("<div class='calc-result'>", unsafe_allow_html=True)
                        st.markdown(f"**📈 Factor de proporción:** `{result['factor_proporcion']:.4f}`")
                        st.markdown(f"**🔴 Valor MÁXIMO proyectado:** `{result['valor_max_proyectado']:.2f} kVA`")
                        st.markdown(f"   → **Día:** {result['dia_max']} | **Hora:** {result['hora_max']}:{int(result['intervalo_max']):02d}")
                        st.markdown(f"**🟢 Valor MÍNIMO proyectado:** `{result['valor_min_proyectado']:.2f} kVA`")
                        st.markdown(f"   → **Día:** {result['dia_min']} | **Hora:** {result['hora_min']}:{int(result['intervalo_min']):02d}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.plotly_chart(plot_weekly_profile(weekly_profile, proj_profile, show_markers=show_markers), use_container_width=True)
                
                # --- Calculadora Total ---
                with calc_tabs[2]:
                    st.markdown("#### Proyección basada en Día Específico")
                    
                    calc_mode = st.radio("Modo de cálculo:", 
                        ["📅 Usar día específico", "🔺 Usar día de MÁXIMA carga", "🔻 Usar día de MÍNIMA carga"],
                        key="calc_mode_total", horizontal=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        calc_power_t = st.number_input("Potencia medida (kVA)", min_value=0.0, value=100.0, step=1.0, key="total_power")
                    
                    use_max = "MÁXIMA" in calc_mode
                    use_min = "MÍNIMA" in calc_mode
                    
                    if not use_max and not use_min:
                        with col2:
                            calc_date_t = st.date_input("Fecha", value=kpis.get('max_date', datetime.now().date()), key="total_date_input")
                        with col3:
                            calc_hour_t = st.number_input("Hora", min_value=0.0, max_value=23.99, value=12.0, step=0.25, key="total_hour")
                        calc_date = calc_date_t
                    else:
                        with col2:
                            calc_hour_t = st.number_input("Hora", min_value=0.0, max_value=23.99, value=12.0, step=0.25, key="total_hour2")
                        calc_date = kpis.get('max_date' if use_max else 'min_date', datetime.now().date())
                        st.info(f"Usando: {calc_date}")
                    
                    if st.button("🔮 Calcular Proyección Total", key="calc_total", type="primary"):
                        result, proj_data, error = calculate_total_projection(
                            df, calc_power_t, calc_date, calc_hour_t, use_max, use_min
                        )
                        
                        if error:
                            st.error(error)
                        else:
                            st.markdown("<div class='calc-result'>", unsafe_allow_html=True)
                            st.markdown(f"**📋 Base de cálculo:** {result['base_calculo']}")
                            st.markdown(f"**📈 Factor de proporción:** `{result['factor_proporcion']:.4f}`")
                            st.markdown(f"**📊 Medida ingresada:** `{result['medida_original']:.2f} kVA` a las `{result['hora_medida']:.2f}h`")
                            st.markdown(f"**🔴 Valor MÁXIMO proyectado:** `{result['valor_max_proyectado']:.2f} kVA` a las `{result['hora_max_proyectado']:.2f}h`")
                            st.markdown(f"**🟢 Valor MÍNIMO proyectado:** `{result['valor_min_proyectado']:.2f} kVA` a las `{result['hora_min_proyectado']:.2f}h`")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.plotly_chart(plot_total_profile(df, None, proj_data, show_markers=show_markers), use_container_width=True)
            
            # === EXPORTAR EXCEL ===
            st.markdown("---")
            st.subheader("📥 Exportar Datos")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Datos', index=False)
                    daily_profile.to_excel(writer, sheet_name='Perfil_Diario', index=False)
                    weekly_profile.to_excel(writer, sheet_name='Perfil_Semanal', index=False)
                buffer.seek(0)
                st.download_button("📊 Descargar Excel", buffer, "analisis_carga.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            with col2:
                kpi_df = pd.DataFrame([kpis]).T.reset_index()
                kpi_df.columns = ['Métrica', 'Valor']
                buffer2 = BytesIO()
                with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
                    kpi_df.to_excel(writer, sheet_name='KPIs', index=False)
                buffer2.seek(0)
                st.download_button("📈 Descargar KPIs", buffer2, "kpis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    else:
        st.info("📁 Sube un archivo Excel para comenzar el análisis")
        
        with st.expander("ℹ️ Instrucciones"):
            st.markdown("""
            **Formato requerido del archivo:**
            - Hoja "DIAGRAMAS": Información del transformador (KVA NOMINAL, SED)
            - Hoja de datos: Mediciones con columnas starttime, STotAvg, U1Avg, I1Avg, etc.
            
            **Funcionalidades:**
            - 📊 Gráficos de potencia con marcadores de máximo y mínimo
            - 🧮 Calculadoras de proyección para cada tipo de perfil
            - 📐 Selector de intervalo (10 min / 1 hora)
            - 📥 Exportación a Excel
            """)


if __name__ == "__main__":
    main()
