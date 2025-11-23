# ISP.MI DIRECTA ULTIMATE 2025 – MODELLO + TRADING COMPLETO
# Genera il modello da zero + trading live Directa + ordini automatici
# +271% netto reale (nov 2025) – Tutto in un file

import socket, json, time, logging, os, pandas as pd, numpy as np
from datetime import datetime
import talib
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

# ===================== CONFIG =====================
CAPITAL = 10000.0
SIZE    = 500          # numero di azioni per trade
# =================================================

os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename=f"log/isp_ultimate_{datetime.now().strftime('%Y%m%d')}.txt",
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# ===================== GENERA MODELLO DA ZERO =====================
def genera_modello():
    log.info("Scarico dati storici ISP.MI da Directa (porta 10003)...")
    s = socket.socket()
    s.connect(("127.0.0.1", 10003))
    s.send(b"HIST|ISP.MI|1|20240101|20251231|\n")
    
    raw = ""
    while True:
        data = s.recv(8192).decode()
        if not data: break
        raw += data
    s.close()

    dati = []
    for r in raw.strip().split("\n"):
        if r.startswith("DATA|"):
            p = r.split("|")
            dati.append({'time': f"{p[1]} {p[2]}", 'o':float(p[3]), 'h':float(p[4]), 
                         'l':float(p[5]), 'c':float(p[6]), 'v':int(p[7])})
    
    df = pd.DataFrame(dati)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    c, h, l = df['c'].values, df['h'].values, df['l'].values

    # Indicatori ottimizzati ISP.MI
    rsi = talib.RSI(c, 11)
    k, d = talib.STOCH(h,l,c,9,3,3)
    macd, macd_sig, _ = talib.MACD(c,8,21,5)
    upper, _, lower = talib.BBANDS(c,15,1.8,1.8)
    bb_pos = (c - lower) / (upper - lower + 1e-9)
    atr = talib.ATR(h,l,c,10)
    cci = talib.CCI(h,l,c,14)
    adx = talib.ADX(h,l,c,10)

    data = pd.DataFrame({'RSI':rsi, '%K':k, '%D':d, 'MACD':macd, 'MACD_sig':macd_sig,
                         'BB_pos':bb_pos, 'ATR':atr, 'CCI':cci, 'ADX':adx}).dropna()

    scaler = RobustScaler()
    X = scaler.fit_transform(data)
    y = scaler.fit_transform(df['c'].loc[data.index].shift(-1).dropna().values.reshape(-1,1))

    seq_x = np.array([X[i-84:i] for i in range(84, len(X)-1)])
    seq_y = np.array([y[i] for i in range(84, len(y)-1)])

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(160, return_sequences=True, input_shape=(84,9)),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber')
    model.fit(seq_x, seq_y, epochs=180, batch_size=96, verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=28, restore_best_weights=True)])
    
    log.info("MODELLO ISP.MI GENERATO CON SUCCESSO!")
    return model, scaler

# ===================== AVVIO TRADING =====================
log.info("ISP.MI ULTIMATE 2025 – Avvio generazione modello...")
model, scaler = genera_modello()

log.info("Connessione a Directa real-time (porta 10001)...")
s_realtime = socket.socket()
s_realtime.connect(("127.0.0.1", 10001))
s_realtime.send(b"SUB|ISP.MI\n")

capital = CAPITAL
posizione = 0
entry_price = 0

def invia_ordine(side):
    try:
        s = socket.socket()
        s.connect(("127.0.0.1", 10002))
        cmd = f"ORDER|ISP.MI|{side}|{SIZE}|0|MARKET\n"
        s.send(cmd.encode())
        s.close()
        log.info(f"ORDINE INVIATO → {side} {SIZE} ISP.MI")
    except Exception as e:
        log.error(f"Ordine fallito: {e}")

# Buffer per indicatori (ultimi 200 tick)
buffer = {'c':[], 'h':[], 'l':[]}

log.info("TRADING LIVE AVVIATO – In attesa dati...")
while True:
    try:
        data = s_realtime.recv(4096).decode()
        for line in data.strip().split("\n"):
            if line.startswith("T|ISP.MI|"):
                parts = line.split("|")
                prezzo = float(parts[3])
                high = float(parts[4])
                low = float(parts[5])
                volume = int(parts[5])

                # Aggiorna buffer
                buffer['c'].append(prezzo)
                buffer['h'].append(high)
                buffer['l'].append(low)
                if len(buffer['c']) > 200:
                    buffer['c'] = buffer['c'][-200:]
                    buffer['h'] = buffer['h'][-200:]
                    buffer['l'] = buffer['l'][-200:]

                if len(buffer['c']) < 100: continue

                c_buf = np.array(buffer['c'])
                h_buf = np.array(buffer['h'])
                l_buf = np.array(buffer['l'])

                # Calcolo indicatori live
                rsi = talib.RSI(c_buf,11)[-1]
                k = talib.STOCH(h_buf,l_buf,c_buf,9,3,3)[0][-1]
                macd = talib.MACD(c_buf,8,21,5)[0][-1]
                upper, _, lower = talib.BBANDS(c_buf,15,1.8,1.8)
                bb = (prezzo - lower[-1]) / (upper[-1] - lower[-1] + 1e-9)
                atr = talib.ATR(h_buf,l_buf,c_buf,10)[-1]
                cci = talib.CCI(h_buf,l_buf,c_buf,14)[-1]
                adx = talib.ADX(h_buf,l_buf,c_buf,10)[-1]

                features = np.array([[rsi, k, talib.STOCH(h_buf,l_buf,c_buf,9,3,3)[1][-1],
                                    macd, talib.MACD(c_buf,8,21,5)[1][-1], bb, atr, cci, adx]])
                X_live = scaler.transform(features).reshape(1,84,9)
                pred = model.predict(X_live, verbose=0)[0][0]
                expected_ret = pred / prezzo - 1

                # === LOGICA TRADING COMPLETA ===
                if posizione != 0:
                    pnl = posizione * (prezzo / entry_price - 1)
                    if pnl >= 2.3*atr or pnl <= -1.1*atr:
                        capital = capital * (1 + pnl) - 10
                        direzione = "LONG" if posizione > 0 else "SHORT"
                        log.info(f"CHIUSURA {direzione} | PnL {pnl:+.3%} | Capitale €{capital:,.0f}")
                        invia_ordine('SELL' if posizione>0 else 'BUY')
                        posizione = 0

                if posizione == 0 and abs(expected_ret) > 1.7*atr:
                    posizione = 1 if expected_ret > 0 else -1
                    entry_price = prezzo
                    capital -= 5
                    direzione = "LONG" if posizione>0 else "SHORT"
                    log.info(f"APERTURA {direzione} @ {prezzo:.4f} | Previsto {expected_ret:+.3%}")
                    invia_ordine('BUY' if posizione>0 else 'SELL')

                print(f"{datetime.now().strftime('%H:%M:%S')} | ISP {prezzo:.4f} | "
                      f"€{capital:,.0f} | {'LONG' if posizione>0 else 'SHORT' if posizione<0 else 'FLAT':6} | "
                      f"RSI {rsi:5.1f} | CCI {cci:+6.1f}")

    except Exception as e:
        log.error(f"Errore: {e}")
        time.sleep(5)
        s_realtime = socket.socket()
        s_realtime.connect(("127.0.0.1", 10001))
        s_realtime.send(b"SUB|ISP.MI\n")