#--------------------------------------------------------------------
#   機能：RRI解析に使用する関数をまとめたもの(mybeat,polymate用)
#
#   使い方："import RRI_analysis"で読み込み，RRI_analysis.func_name()で使用
#--------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, fftpack, signal
import matplotlib as mpl
import matplotlib.patheffects as path_effects

#############################################################################
#                                  Main                                     #
#############################################################################
# time_index: 出力するデータの時間軸．rriの畳み込み和[msec]を1000で割って解析期間[sec]を算出
def rri_analysis(time_index, rri, outputdir, samplingrate):
    # 周波数解析の前にリサンプリング
    resampling_freq = 10
    resampling_rri = resample(time_index, rri, resampling_freq)
        
    # wavelet変換を用いた時間周波数解析
    tm, LFHF_T, resampling_freq = wavelet(resampling_freq, resampling_rri, outputdir)
    
    # 各時間幅ごとにHR,nHF,LF/HFを算出
    timewidths = [60, 900] # 時間幅[sec]（細かい変動を見るための1minと，血糖値と同じ時間点数15minの2種）
    for timewidth in timewidths:
        HR = caluculateHR(tm, resampling_freq, resampling_rri, samplingrate, timewidth)
        #fft(time_index, resampling_freq, resampling_rri, outputdir, timewidth)  #FFTを用いた時間周波数解析
        time, nHF, LFHF = devide_LFHF(tm, LFHF_T, timewidth, resampling_freq)  #waveletで計算したLFHFを分割
        
        outputdf = pd.DataFrame(
            data = [time, HR, nHF, LFHF],
            index = ['time[min]', 'HR', 'nHF', 'LFHF'],
        )
        pd.DataFrame(outputdf.T).to_csv(outputdir+"result_per"+str(int(timewidth/60))+"min.csv", index=False)

    # 任意関数
    #instantaneous_HR(time_index,rri,outputdir)  #瞬時心拍数
    #lorenzplot(time_index,rri,outputdir)   #ローレンツプロット


#############################################################################
#                                  前処理                                    #
#############################################################################

# 外れ値除去＆スプライン補間
def outlierremoval(time_index, rri, samplingrate, MinHR, MaxHR):
    # 外れ値除去
    for i in range(len(rri)-1):
        # HR が MinHR〜MaxHR の範囲外のrriを除去
        if 60*samplingrate/MinHR < rri[i] or rri[i] < 60*samplingrate/MaxHR:
            rri[i] = np.nan

    # 3次スプライン補間
    df = pd.DataFrame(data=rri, index=time_index, columns=["rri"], dtype=np.float32)
    df.interpolate('spline',order = 3,inplace = True,limit_direction='both')
    df = df.astype('int')
    rri = df.values
    rri = [x for row in rri for x in row]

    return rri

# リサンプリングを実行する関数
def resample(time_index, rri, resampling_freq):
    new_t = np.arange(0, int(time_index[-1]), 1/resampling_freq)  # 等間隔の新しい時間軸の設定
    spline_func = interpolate.interp1d(time_index, rri, fill_value="extrapolate")
    resampling_rri = spline_func(new_t).round(6)

    return resampling_rri


#############################################################################
#                                   FFT                                     #
#############################################################################
def fft(time_index, resampling_freq, resampling_rri, outputdir, timewidth):
    LFHF = []
    time = []

    # FFTの実行とHF,LFの算出
    for width in range(int(time_index[-1]/timewidth)):
        target_rri = resampling_rri[timewidth*width*resampling_freq:timewidth*(width+1)*resampling_freq]

        window = signal.hamming(timewidth*resampling_freq)  # ハミング窓関数を使用
        windowed_rri = target_rri * window

        freq_list = fftpack.fftfreq(len(target_rri), d=1/resampling_freq)
        y_fft = fftpack.fft(windowed_rri)
        pidxs = np.where(freq_list > 0)
        freqs, power = freq_list[pidxs], np.abs(y_fft)[pidxs]
        freq = list(zip(freqs, power))

        LF = 0
        HF = 0

        for freq_i in range(len(freqs)):
            if freq[freq_i][0] > 0.04 and freq[freq_i][0] < 0.15:
                LF += freq[freq_i][1]
            elif freq[freq_i][0] > 0.15 and freq[freq_i][0] < 0.4:
                HF += freq[freq_i][1]
        
        time.append(timewidth*width/60 + 7.5)
        LFHF.append([HF/(LF+HF), LF/HF])
    
    # データ保存
    df = pd.DataFrame(data=LFHF, index=time, columns=["nHF","LFHF"])
    pd.DataFrame(df).to_csv(outputdir+"LFHF_per"+str(int(timewidth/60))+"min.csv")


#############################################################################
#                                  wavelet                                  #
#############################################################################
def wavelet(resampling_freq, resampling_rri, outputdir):    
    dt = 1/resampling_freq
    tms = 0     #開始時間
    tme = len(resampling_rri)/resampling_freq   #終了時間
    tm = np.arange(tms, tme, dt)

    # n_cwt をlen(resampling_rri)よりも大きな2のべき乗の数になるように設定
    n_cwt = int(2**(np.ceil(np.log2(len(resampling_rri)))))

    # パラメータ定義
    dj = 0.125
    omega0 = 6.0
    s0 = 2.0*dt
    J = int(np.log2(n_cwt*dt/s0)/dj)

    # スケール
    s = s0*2.0**(dj*np.arange(0, J+1, 1))

    # n_cwt個のデータになるようにゼロパディングして，DC成分を除く
    x = np.zeros(n_cwt)
    x[0:len(resampling_rri)] = resampling_rri[0:len(resampling_rri)] - np.mean(resampling_rri)

    omega = 2.0*np.pi*np.fft.fftfreq(n_cwt, dt)

    # FFTを使って離散ウェーブレット変換する
    X = np.fft.fft(x)
    cwt = np.zeros((J+1, n_cwt), dtype=complex) 

    Hev = np.array(omega > 0.0)
    for j in range(J+1):
        Psi = np.sqrt(2.0*np.pi*s[j]/dt)*np.pi**(-0.25)*np.exp(-(s[j]*omega-omega0)**2/2.0)*Hev
        cwt[j, :] = np.fft.ifft(X*np.conjugate(Psi))

    s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi) 
    freq_cwt = s_to_f / s
    cwt = cwt[:, 0:len(resampling_rri)]
    
    #LFHFの計算
    LFHF_T = LFHFcalculator(np.log10(np.abs(cwt)**2), freq_cwt)
    df = pd.DataFrame(data=LFHF_T, index=tm, columns=["nHF","LFHF"])
    pd.DataFrame(df).to_csv(outputdir+"wavelet.csv")    # データ保存

    return tm, LFHF_T, resampling_freq


# LFHFの計算
def LFHFcalculator(cwtmatr, frequencies):
    # 指定の周波数帯域の行の番号
    LFcom = []
    HFcom = []

    # 特定の周波数帯域の行列
    LFfre = []
    HFfre = []
    
    # 指定の周波数帯域のパワーの合計（時間ごと）
    LF = 0 
    HF = 0

    for i in range(len(frequencies)-1):
        if frequencies[i] > 0.04 and frequencies[i] < 0.15:
            LFcom.append(i)
        if frequencies[i] > 0.15 and frequencies[i] < 0.4:
            HFcom.append(i)
    for i in LFcom:
        LFfre.append(abs(cwtmatr[i,:]))
    for i in HFcom:
        HFfre.append(abs(cwtmatr[i,:]))
    LF = np.sum(LFfre, axis=0)
    HF = np.sum(HFfre, axis=0)
    
    LFHF = LF/HF
    nHF = HF / (LF + HF)
    LFHF_T = np.array([nHF, LFHF]).T

    return LFHF_T


def devide_LFHF(tm, LFHF_T, timewidth, resampling_freq):
    nHF_data = LFHF_T[:,0]
    LFHF_data = LFHF_T[:,1]

    time = []
    nHF = []
    LFHF = []

    for width in range(int(tm[-1]/timewidth)):
        nHF_dur = np.median(nHF_data[timewidth*width*resampling_freq:timewidth*(width+1)*resampling_freq])
        LFHF_dur = np.median(LFHF_data[timewidth*width*resampling_freq:timewidth*(width+1)*resampling_freq])

        time.append(timewidth*width/60 + timewidth/120)
        nHF.append(nHF_dur)
        LFHF.append(LFHF_dur)

    return time, nHF, LFHF


#############################################################################
#                                Sub_function                               #
#############################################################################
def caluculateHR(time_index, resampling_freq, resampling_rri, samplingrate, timewidth):
    HR = []

    # HRの計算
    for width in range(int(time_index[-1]/timewidth)):
        target_rri = resampling_rri[timewidth*width*resampling_freq:timewidth*(width+1)*resampling_freq]
        HR.append(60*samplingrate/np.average(target_rri))
    
    return HR

# 瞬時心拍数のグラフ
def instantaneous_HR(time_index, rri,outputdir, samplingrate):
    hr = []
    for i in range(len(rri)):
        hr.append(60*samplingrate / rri[i])
    saveplot("HR", time_index, hr, "Time [s]", "HR [bpm]",outputdir)

# ローレンツプロット
def lorenzplot(time_index, rri, outputdir):
    xdata = rri[0:-2] #nデータ
    ydata = rri[1:-1] #n+1データ

    plt.figure()
    plt.scatter(xdata, ydata)
    plt.xlabel("RRI(n) [ms]")
    plt.ylabel("RRI(n+1) [ms]")
    plt.savefig(outputdir+"lorenzplot.png")
    #plt.show()

    dif_rri = xdata - ydata
    sum_rri = xdata + ydata

    SD1 = np.std(dif_rri) / np.sqrt(2)
    SD2 = np.std(sum_rri) / np.sqrt(2)
    print("SD1 = ", round(SD1,2))
    print("SD2 = ", round(SD2,2))


#############################################################################
#                                   Output                                  #
#############################################################################
# グラフ出力と保存
def saveplot(name, x, y, xlabel, ylabel,outputdir):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0,3000)
    plt.tick_params(labelbottom=False)
    plt.savefig(outputdir+name+".png") 
    plt.show()

# ２つグラフを表示・保存
def saveplot2(name, x, y, xlabel, ylabel, x2, y2, xlabel2, ylabel2, outputdir):
    fig = plt.figure()
    a1 = fig.add_subplot(2,1,1) #(縦分割数、横分割数、ポジション)
    a1.plot(x, y)
    a1.set_xlabel(xlabel)
    a1.set_ylabel(ylabel)
    a1.tick_params(labelbottom=False)
    a1.set_ylim(0,3000)
    a1.set_title("original_rri")

    a2 = fig.add_subplot(2,1,2)
    a2.plot(x2, y2)
    a2.set_xlabel(xlabel2)
    a2.set_ylabel(ylabel2)
    a2.tick_params(labelbottom=False)
    a2.set_ylim(0,3000)
    a2.set_title("outlierremoval_rri")

    plt.savefig(outputdir+name+".png") 
    #plt.show()
