#--------------------------------------------------------------------
#   機能：mybeatで計測したrriのrawdataから心拍変動を解析s
#
#   使い方："import RRI_analysis"で読み込み，RRI_analysis.func_name()で使用
#--------------------------------------------------------------------

import numpy as np
import pandas as pd
from itertools import accumulate
import datetime
import RRI_analysis


def main():
    # ファイル名と開始時点
    file = {'S003_1207_1':2,'S003_1212_2':35,'S003_1221_3':15,'S004_1212_2':35,'S004_1220_1':2,'S004_1227_3':5}

    for ID, min in file.items():
        analysis_mybeat(ID, min)
    

def analysis_mybeat(file_ID, starttime_min):
    ### 解析に使用するデータを用意
    samplingrate = 1000

    # 入力と出力ディレクトリ tsujiryouhei
    csv_path = '/Users/tsujiryouhei/Dropbox/1_血糖値/3_実験結果/プレ実験/'+file_ID+'/rawdata.csv'
    outputdir = '/Users/tsujiryouhei/Dropbox/1_血糖値/3_実験結果/プレ実験/'+file_ID+'/解析結果/' 
    print(file_ID)

    time_index, data = dataread(csv_path, starttime_min, file_ID, outputdir, samplingrate) 
    rri = data[:,0]
    
    ### 上記で用意したデータを用いて解析
    RRI_analysis.rri_analysis(time_index, rri, outputdir, samplingrate)    


def dataread(csv_path, starttime_min, file_ID, output, samplingrate):
    # データ読み込み
    csv= pd.read_csv(filepath_or_buffer=csv_path, encoding="utf-8", sep=",", header=5, index_col=0) 
   
    datastart = []

    for x in range(len(csv)):
        try:
            dte = datetime.datetime.strptime(csv.index[x], '%Y/%m/%d %H:%M:%S.%f')
        except: 
            dte = datetime.datetime.strptime(csv.index[x], '%M:%S.%f')

        if dte.minute == starttime_min or dte.minute == starttime_min+3:
            datastart.append(x)

    print(f"{datastart[0]}番目のデータから抽出")    #開始直後のデータ
    print(f"{datastart[-1]}番目のデータから抽出")   #終了直後のデータ
    data = csv[datastart[0]:datastart[-1]+1].values  #実験中のデータ
    
    time_index = [t for t in accumulate(data[:,0]/1000)]    #時間軸の作成 [sec]
    
    outlierremoval_rri = RRI_analysis.outlierremoval(time_index, data[:,0].copy(), samplingrate, MinHR=50, MaxHR=150)  #外れ値除去＆補間
    
    # rriの可視化と外れ値除去
    RRI_analysis.saveplot2(file_ID+"_rri", time_index, data[:,0], "Time [s]", "rri [ms]", time_index, outlierremoval_rri, "Time [s]", "rri [ms]",outputdir=output)

    data[:,0] = outlierremoval_rri

    return time_index, data


if __name__ == '__main__':
    main()