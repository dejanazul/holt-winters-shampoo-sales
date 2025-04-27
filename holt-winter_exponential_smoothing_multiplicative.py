import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./shampoo_sales.csv")
plt.figure(figsize=(12, 6))
plt.scatter(df["Month"], df["Sales"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# level l0 = rata rata penjualan tahun pertama
sales_first_year = df["Sales"][:12]
l0 = sales_first_year.mean()

# t0 = (rata" penjualan tahun 2 - rata" penjualan tahun 1) / 12
sales_second_year = df["Sales"][12:24]
mean_sales_second_year = sales_second_year.mean()
t0 = (mean_sales_second_year - l0) / 12

# musiman awal (s1, s2, s3, ... s12) = sales bulan 1 - 12 /
sn = [(n / l0) for n in sales_first_year]
np.array(sn)

# parameter smoothing = alpha, beta, gamma
alpha = 0.2
beta = 0.2
gamma = 0.15
snt_for_lt = 0
snt_for_st = 0
snt_for_ft = 0


def Lt(x, sn, Lts, Tt, t):
    global snt_for_lt
    if t == 0:
        Lt = (alpha * (x[t] / sn[t])) + ((1 - alpha) * (l0 + t0))
    elif t <= 11:
        Lt = (alpha * (x[t] / sn[t])) + ((1 - alpha) * (Lts[t - 1] + Tt[t - 1]))
    elif t <= 23:
        Lt = (alpha * (x[t] / sn[snt_for_lt])) + (
            (1 - alpha) * (Lts[t - 1] + Tt[t - 1])
        )
        snt_for_lt += 1
    return Lt


def Tt(Lt, Tts, t):
    if t == 0:
        Tt = (beta * (Lt[t] - l0)) + ((1 - beta) * t0)
    elif t <= 23:
        Tt = (beta * (Lt[t] - Lt[t - 1])) + ((1 - beta) * Tts[t - 1])
    return Tt


def St(x, Lt, sn, t):
    global snt_for_st
    if t <= 11:
        St = (gamma * (x[t] / Lt[t])) + ((1 - gamma) * sn[t])
    elif t <= 23:
        St = (gamma * (x[t] / Lt[t])) + ((1 - gamma) * sn[snt_for_st])
        snt_for_st += 1
    return St


def Ft(Lt, Tt, sn, t):
    global snt_for_ft
    if t <= 10:
        Ft = (Lt[t] + 1 * Tt[t]) * sn[t + 1]
    elif t % 12 == 0:
        Ft = (Lt[t] + 1 * Tt[t]) * sn[0]
    elif t <= 23:
        if t == 23:
            return
        Ft = (Lt[t] + 1 * Tt[t]) * sn[snt_for_ft + 1]
    return Ft


# lt, tt, st mulai dari tahun ke2
x = df["Sales"][12:].to_numpy()
Lts = []
Tts = []
Sts = []
Fts = []
len(Fts)
for t in range(len(x)):
    Lts.append(Lt(x, sn, Lts, Tts, t))
    Tts.append(Tt(Lts, Tts, t))
    Sts.append(St(x, Lts, sn, t))
    Fts.append(Ft(Lts, Tts, sn, t))

# evaluasi model
x = np.delete(x, 0)
Fts = np.array(Fts)
Fts = np.delete(Fts, 23)

loss = []
for i in range(len(x)):
    loss.append(x[i] - Fts[i])

loss_sum = sum(loss)
loss_sum_squared = sum([np.square(i) for i in loss])
percentage_errors = []
for i in range(len(x)):
    if x[i] != 0:
        percentage_error = abs((x[i] - Fts[i]) / x[i])
        percentage_errors.append(percentage_error)
mape = sum(percentage_errors) / len(percentage_errors) * 100 

mean_absolute_error = loss_sum / len(loss)
mean_squared_error = loss_sum_squared / len(loss)

print("Holt Winter Multiplicative Model Evaluation")
print(f"mean absolute error = {mean_absolute_error:.3f}")
print(f"mean squared error = {mean_squared_error:.3f}")
print(f"Mean Absolute Percentage Error (MAPE) = {mape:.2f}%")

# Result visualization
plt.figure(figsize=(12, 6))
plt.plot(df["Month"], df["Sales"], label="Actual Sales", marker='o')
forecast_months = df["Month"][13:36]  # Adjusted to match Fts length
plt.plot(forecast_months, Fts, label="Predicted Sales", linestyle='--', marker='x')

# Formatting
plt.title("Actual vs Predicted Shampoo Sales (Holt-Winters Multiplicative Model)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save png
plt.savefig('holt_winters_result.png', dpi=300)
plt.show()