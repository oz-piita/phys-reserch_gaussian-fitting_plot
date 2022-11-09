# ピーク数可変、バックグラウンド変数多項式可変、パラメタ固定も容易なガウシアンフィッティングコードの備忘録

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from scipy.optimize import curve_fit
import math

samplename = "Ni"
filename = "./2022Ni25smp.txt"
wanna_save_fig = True
wanna_use_fixed_bkg_param = False        # Trueの場合、n1とn0を固定して使います
peak_cnt = 2       # ガウシアンの数
start = 1840        # x軸の両端
end = 2400

a0= 1.6613791311481552
b0= 2077.0326976476194
c0= 35
a1= 0.33
b1= 2152
c1= 22
n1 =  -0.00029
n0 =  1.488

param_ini = [a0, b0, c0, a1, b1, c1, n1, n0]
# パラメタを増減させる際は【param_iniの中身を編集し、peak_cntの数を設定】する
# a1b1c1の3つでガウスピーク1つ分。n1,n0などはバックグラウンド多項式の係数。増やせる。

def func(x, *params):           # ガウシアン
    y_list = []
    for i in range(peak_cnt):
        amp = params[3*i]
        ctr = params[3*i+1]
        wid = params[3*i+2]
        # if i == 0:                          # ピークパラメタを固定したいときはこの辺で割り込む
        #     amp= a0
        #     wid = c0
        # elif i==1:
        #     amp = a1
        #     ctr = b1
        #     wid = c1
        y = np.zeros_like(x)
        y = y + amp * np.exp( - ((x - ctr)/wid)**2)
        y_list.append(y)
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i
    y_sum = plus_bkg(params, x, y_sum)
    return y_sum

def main():
    raw_data = open_text_file(filename)
    xx, yy = cut_strtend(raw_data, start, end)      # データのx軸レンジを切り出す

    popt, pcov = curve_fit(func, xx, yy, p0 = param_ini)        # solve

    output_param(popt)

    baseline = plus_bkg(popt, xx, np.zeros_like(xx))
    fitting = func(xx, *popt)          # solve後のパラメータからフィッティングカーブを計算
    peaks = calc_each_peaks(xx, *popt) + baseline   # ガウシアンそれぞれのy。np配列2つ分

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(xx, yy,s=10, color = "black", label = samplename)          # 生データプロット
    plt.plot(xx, fitting , ls='-', c='red', lw=1, label = "fitting carve")   # トータルフィッティングプロット
    plt.plot(xx,baseline,color="gray",linestyle="dashed",label = "background")
    for n,i in enumerate(peaks):
        plt.plot(xx,i)
    #     plt.fill_between(xx, i, facecolor=cm.rainbow(n/len(peaks)), alpha=0.6)
    plt.fill_between(xx, fitting, baseline, facecolor="red", alpha=0.2)

    calc_residual_sumsq(yy,fitting)
    plt.xlabel('Wave Number (/cm)',fontsize = 12)
    plt.ylabel('Absorbance(arb.units)',fontsize = 12)
    plt.ylim(0,np.max(yy)+0.1)
    plt.legend()
    plt.tight_layout()
    if wanna_save_fig:
        plt.savefig('./fig/curvefit2'+samplename+'.png', dpi=120)
    plt.show()

    return

def output_param(params):
    print("sample,start,end:",samplename, start, end)
    for i in range(peak_cnt):
        # print("a"+str(i)+"=", params[3*i], params[3*i+1], params[3*i+2])
        print("a"+str(i)+"=", params[3*i])
        print("b"+str(i)+"=", params[3*i+1])
        print("c"+str(i)+"=", params[3*i+2])
    if wanna_use_fixed_bkg_param:
       print("n1 = ",n1)
       print("n0 = ",n0)
    elif not wanna_use_fixed_bkg_param:
        bkg_param_cnt = len(params) - peak_cnt*3
        for i in range(bkg_param_cnt):
            print("n"+str(bkg_param_cnt-1-i)+" = ", params[-bkg_param_cnt+i], end = ", ")
    return

def calc_residual_sumsq(yy, fitting):       # 残差二乗和Residual Sum of Squareを表示
    residual_arr = (yy - fitting)**2
    print("Residual Sum of Square:",np.sum(residual_arr))
    return

def plus_bkg(params,x, y_sum):
    if wanna_use_fixed_bkg_param:
        return y_sum + n1 *x + n0                   # バックグラウンドを目算で固定する場合分け
    bkg_param_cnt = len(params) - peak_cnt * 3      # 3はピークあたりのパラメタ数
    for i in range(bkg_param_cnt):          # 整式を足す.何次式でも.
        y_sum += params[-(i+1)] * x ** i
    return y_sum

def calc_each_peaks(x, *params):
    y_list = []
    for i in range(peak_cnt):
        amp = params[3*i]
        ctr = params[3*i+1]
        wid = params[3*i+2]

        y = np.zeros_like(x)
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)
    return y_list

class Ftir():
    def __init__(self,wavenum,absorb):
        self.wavenum = wavenum   # float
        self.absorb  = absorb    # float

def cut_strtend(raw_data, stt, ed):
    xx = []
    yy = []
    for lin in raw_data:
        if ed < lin.wavenum:
            break
        if stt < lin.wavenum and lin.wavenum < ed:
            cut = lin.absorb
            xx.append(lin.wavenum)
            yy.append(cut)
    return np.array(xx), np.array(yy)

def open_text_file(raw_data_file_name):
    rst = []
    ln_cnt = 0
    with open(raw_data_file_name) as f:
        for line in f:  # read line by line.
            indb = line.split("	")
            if 19 <= ln_cnt:  
                aa = Ftir(float(indb[0]),float(indb[1]))
                if  3000 <= aa.wavenum  :                   # 3000以上はこの時点で切る
                    break
                rst.append(aa)
            ln_cnt += 1
    return rst
                
if __name__ == "__main__":
    main()
